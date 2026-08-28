//===- SMatMulMatmulToBankSSAPatterns.cpp - f32 matmul -> BankMatrix
//-------===//
//
// Bank unit: all matrix dimensions are multiples of 16.
// One activation chunk is packed and quantized once before its N panels run.
//
//===----------------------------------------------------------------------===//

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

constexpr int64_t kLane = 16;
constexpr int64_t kBankDepth = 1024;
constexpr int64_t kMmioBytes = 5 * 1024;

struct OutputPack {
  Value value;
  int64_t m0;
  int64_t mChunk;
  int64_t n0;
  int64_t nChunk;
};

class MatrixMatmulToBankSSAPattern : public OpRewritePattern<SMatMulMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SMatMulMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto aTy = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto bTy = dyn_cast<MemRefType>(op.getBMemArray().getType());
    auto cTy = dyn_cast<MemRefType>(op.getCMemArray().getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() ||
        !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");

    Type elem = aTy.getElementType();
    if (isa<FloatType>(elem) && bTy.getElementType() == elem &&
        cTy.getElementType() == elem) {
      int64_t M = aTy.getShape()[0], K = aTy.getShape()[1];
      int64_t Kb = bTy.getShape()[0], N = bTy.getShape()[1];
      if (K != Kb || cTy.getShape()[0] != M || cTy.getShape()[1] != N)
        return op.emitError("matmul shape mismatch");
      b.create<linalg::MatmulOp>(
          loc, ValueRange{op.getAMemArray(), op.getBMemArray()},
          ValueRange{op.getCMemArray()});
      b.eraseOp(op);
      return success();
    }

    if (!aTy.getElementType().isF32() || !bTy.getElementType().isInteger(8) ||
        !cTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");

    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t M = aTy.getShape()[0], K = aTy.getShape()[1];
    int64_t Kb = bTy.getShape()[0], N = bTy.getShape()[1];
    if (K != Kb || cTy.getShape()[0] != M || cTy.getShape()[1] != N)
      return op.emitError("matmul shape mismatch");
    if (M % kLane || N % kLane || K % kLane)
      return op.emitError("SMatMul requires M/N/K to be 16-aligned");
    if (M <= 0 || K <= 0 || N <= 0)
      return op.emitError("M/N/K must be positive");
    const int64_t outputGroups =
        buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW;
    if (outputGroups <= 0 || outputGroups > 4 || 4 % outputGroups)
      return op.emitError("SMatMulBall outBW must divide four result blocks");
    const int64_t outputRounds = 4 / outputGroups;
    int64_t requiredDwBytes = perChannel ? N * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");
    const int64_t kTiles = K / kLane;
    const int64_t aRowsPerMTile = K;
    const int64_t maxMChunk = (kBankDepth / aRowsPerMTile) * kLane;
    if (maxMChunk == 0)
      return op.emitError("K exceeds activation-bank packing capacity");

    SmallVector<Value> activationPacks;
    SmallVector<Value> weightPacks;
    SmallVector<OutputPack> outputPacks;
    Value aF = allocBank(b, loc, 1, 4);
    Value aI = allocBank(b, loc, 1, 1);
    Value bI = allocBank(b, loc, 1, 1);
    Value cB = allocBank(b, loc, 1, outputGroups);
    Value cF = allocBank(b, loc, 1, outputGroups);

    // Quantize each activation chunk once, then reuse its Da and INT8 bank
    // across every output panel. The panel loop only changes the weight/Dw
    // slice and the destination tile.
    for (int64_t m0 = 0; m0 < M;) {
      int64_t mChunk = std::min(maxMChunk, M - m0);
      if (mChunk > 4095)
        return op.emitError("matmul M chunk exceeds matrix cfg field");

      const int64_t aRows = mChunk / kLane * aRowsPerMTile;
      Value daAddr = createI64Const(b, loc, 0);
      auto aPackType = MemRefType::get({aRows, kLane}, b.getF32Type());
      Value aPack = b.create<memref::AllocOp>(loc, aPackType);
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
      Value mUpper = b.create<arith::ConstantIndexOp>(loc, mChunk / kLane);
      Value kUpper = b.create<arith::ConstantIndexOp>(loc, kTiles);
      Value mBase = b.create<arith::ConstantIndexOp>(loc, m0);
      Value kTilesValue = b.create<arith::ConstantIndexOp>(loc, kTiles);
      auto mLoop = b.create<scf::ForOp>(loc, zero, mUpper, one);
      b.setInsertionPointToStart(mLoop.getBody());
      Value m = mLoop.getInductionVar();
      auto kLoop = b.create<scf::ForOp>(loc, zero, kUpper, one);
      b.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      auto rowLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
      b.setInsertionPointToStart(rowLoop.getBody());
      Value row = rowLoop.getInductionVar();
      auto columnLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
      b.setInsertionPointToStart(columnLoop.getBody());
      Value column = columnLoop.getInductionVar();
      Value sourceRow = b.create<arith::AddIOp>(
          loc, mBase,
          b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, m, sixteen),
                                  row));
      Value sourceColumn = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, k, sixteen), column);
      Value tile = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, m, kTilesValue), k);
      Value packedRow = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, tile, sixteen), row);
      Value value = b.create<memref::LoadOp>(
          loc, op.getAMemArray(), ValueRange{sourceRow, sourceColumn});
      b.create<memref::StoreOp>(loc, value, aPack,
                                ValueRange{packedRow, column});
      b.setInsertionPointAfter(mLoop);
      activationPacks.push_back(aPack);
      Value aL = mvinBank(b, loc, aPack, aF, aRows, 1);
      Value aQ = b.create<BankFp2IntOp>(loc, aI.getType(), aL, aI,
                                        createI64Const(b, loc, aRows), daAddr);

      for (int64_t n0 = 0; n0 < N; n0 += kLane) {
        int64_t dwAddr = dwAddrBase + (perChannel ? n0 * 4 : 0);
        Value dwAddrValue = createI64Const(b, loc, dwAddr);
        uint64_t cfg = matrixRs2((uint64_t)mChunk, 16, (uint64_t)K);
        auto bPackType = MemRefType::get({K, kLane}, bTy.getElementType());
        Value bPack = b.create<memref::AllocOp>(loc, bPackType);
        Value bZero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value bOne = b.create<arith::ConstantIndexOp>(loc, 1);
        Value bK = b.create<arith::ConstantIndexOp>(loc, K);
        Value bN = b.create<arith::ConstantIndexOp>(loc, n0);
        Value bSixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
        auto bRowLoop = b.create<scf::ForOp>(loc, bZero, bK, bOne);
        b.setInsertionPointToStart(bRowLoop.getBody());
        Value bRow = bRowLoop.getInductionVar();
        auto bColumnLoop = b.create<scf::ForOp>(loc, bZero, bSixteen, bOne);
        b.setInsertionPointToStart(bColumnLoop.getBody());
        Value bColumn = bColumnLoop.getInductionVar();
        Value bSourceColumn = b.create<arith::AddIOp>(loc, bN, bColumn);
        Value bValue = b.create<memref::LoadOp>(
            loc, op.getBMemArray(), ValueRange{bRow, bSourceColumn});
        b.create<memref::StoreOp>(loc, bValue, bPack,
                                  ValueRange{bRow, bColumn});
        b.setInsertionPointAfter(bRowLoop);
        weightPacks.push_back(bPack);
        Value bL = mvinBank(b, loc, bPack, bI, aRowsPerMTile);
        Value cO = createBankSMatMul(b, loc, cB.getType(), aQ, bL, cB,
                                     createI64Const(b, loc, (int64_t)cfg));
        Value fp = perChannel
                       ? b.create<BankInt2FpChannelOp>(
                              loc, cF.getType(), cO, cF,
                              createI64Const(b, loc, outputRounds * mChunk),
                              daAddr, dwAddrValue)
                             .getResult()
                       : b.create<BankInt2FpTensorOp>(
                              loc, cF.getType(), cO, cF,
                              createI64Const(b, loc, outputRounds * mChunk),
                              daAddr, dwAddrValue)
                             .getResult();
        auto cPackType = MemRefType::get(
            {outputRounds * mChunk, outputGroups * 4}, b.getF32Type());
        Value cPack = b.create<memref::AllocOp>(loc, cPackType);
        mvoutBank(b, loc, cPack, fp, outputRounds * mChunk, 1);
        outputPacks.push_back({cPack, m0, mChunk, n0, kLane});
      }
      m0 += mChunk;
    }

    releaseBank(b, loc, aF);
    releaseBank(b, loc, aI);
    releaseBank(b, loc, bI);
    releaseBank(b, loc, cB);
    releaseBank(b, loc, cF);

    b.create<FenceOp>(loc);
    for (const OutputPack &pack : outputPacks) {
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      Value rounds = b.create<arith::ConstantIndexOp>(loc, outputRounds);
      Value groups = b.create<arith::ConstantIndexOp>(loc, outputGroups);
      Value four = b.create<arith::ConstantIndexOp>(loc, 4);
      Value mUpper = b.create<arith::ConstantIndexOp>(loc, pack.mChunk);
      Value nUpper = b.create<arith::ConstantIndexOp>(loc, pack.nChunk);
      Value mBase = b.create<arith::ConstantIndexOp>(loc, pack.m0);
      Value nBase = b.create<arith::ConstantIndexOp>(loc, pack.n0);
      auto mLoop = b.create<scf::ForOp>(loc, zero, mUpper, one);
      b.setInsertionPointToStart(mLoop.getBody());
      Value m = mLoop.getInductionVar();
      auto nLoop = b.create<scf::ForOp>(loc, zero, nUpper, one);
      b.setInsertionPointToStart(nLoop.getBody());
      Value n = nLoop.getInductionVar();
      Value block = b.create<arith::DivUIOp>(loc, n, four);
      Value packedRow =
          b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, m, rounds),
                                  b.create<arith::DivUIOp>(loc, block, groups));
      Value packedColumn = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(
              loc, b.create<arith::RemUIOp>(loc, block, groups), four),
          b.create<arith::RemUIOp>(loc, n, four));
      Value value = b.create<memref::LoadOp>(
          loc, pack.value, ValueRange{packedRow, packedColumn});
      Value row = b.create<arith::AddIOp>(loc, mBase, m);
      Value column = b.create<arith::AddIOp>(loc, nBase, n);
      b.create<memref::StoreOp>(loc, value, op.getCMemArray(),
                                ValueRange{row, column});
      b.setInsertionPointAfter(mLoop);
    }
    for (Value pack : activationPacks)
      b.create<memref::DeallocOp>(loc, pack);
    for (Value pack : weightPacks)
      b.create<memref::DeallocOp>(loc, pack);
    for (const OutputPack &pack : outputPacks)
      b.create<memref::DeallocOp>(loc, pack.value);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populateSMatMulBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatrixMatmulToBankSSAPattern>(patterns.getContext());
}
