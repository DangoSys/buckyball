//===- SMatMulMatmulToBankSSAPatterns.cpp - f32 matmul -> BankMatrix
//-------===//
//
// Bank unit: M_chunk <= bankDepth, K/N edge tiles are explicitly zero-padded.
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
#include "Utils/BankUtils.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kLane = 16;
constexpr int64_t kBankDepth = 1024;
constexpr int64_t kMmioBytes = 5 * 1024;

static int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

// SMatMul consumes A in [M-tile][K-tile][lane] order. The explicit pack
// provides zero-padded edge tiles without changing the activation scale.
static Value packActivation(OpBuilder &b, Location loc, Value source,
                            int64_t m0, int64_t mChunk, int64_t k) {
  const int64_t mTiles = ceilDiv(mChunk, kLane);
  const int64_t kTiles = ceilDiv(k, kLane);
  const int64_t rows = mTiles * kTiles * kLane;
  auto packType = MemRefType::get({rows, kLane}, b.getF32Type());
  Value pack = b.create<memref::AllocOp>(loc, packType);
  Value f0 =
      b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  b.create<linalg::FillOp>(loc, f0, pack);

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
  Value mUpper = b.create<arith::ConstantIndexOp>(loc, mTiles);
  Value kUpper = b.create<arith::ConstantIndexOp>(loc, kTiles);
  Value mBase = b.create<arith::ConstantIndexOp>(loc, m0);
  Value mEnd = b.create<arith::ConstantIndexOp>(loc, m0 + mChunk);
  Value kEnd = b.create<arith::ConstantIndexOp>(loc, k);
  Value kTilesV = b.create<arith::ConstantIndexOp>(loc, kTiles);

  auto mLoop = b.create<scf::ForOp>(loc, zero, mUpper, one);
  b.setInsertionPointToStart(mLoop.getBody());
  Value mt = mLoop.getInductionVar();
  auto kLoop = b.create<scf::ForOp>(loc, zero, kUpper, one);
  b.setInsertionPointToStart(kLoop.getBody());
  Value kt = kLoop.getInductionVar();
  auto rLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(rLoop.getBody());
  Value r = rLoop.getInductionVar();
  auto cLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(cLoop.getBody());
  Value c = cLoop.getInductionVar();

  Value srcRow = b.create<arith::AddIOp>(
      loc, mBase,
      b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, mt, sixteen),
                              r));
  Value srcCol = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, kt, sixteen), c);
  Value tile = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, mt, kTilesV), kt);
  Value dstRow = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, tile, sixteen), r);
  Value rowValid =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, srcRow, mEnd);
  Value colValid =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, srcCol, kEnd);
  Value valid = b.create<arith::AndIOp>(loc, rowValid, colValid);
  auto load = b.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
  b.setInsertionPointToStart(&load.getThenRegion().front());
  Value value =
      b.create<memref::LoadOp>(loc, source, ValueRange{srcRow, srcCol});
  b.create<memref::StoreOp>(loc, value, pack, ValueRange{dstRow, c});
  b.setInsertionPointAfter(mLoop);
  return pack;
}

static Value packWeight(OpBuilder &b, Location loc, Value source, int64_t n0,
                        int64_t k, int64_t nChunk) {
  const int64_t kTiles = ceilDiv(k, kLane);
  const int64_t rows = kTiles * kLane;
  auto sourceType = cast<MemRefType>(source.getType());
  auto packType = MemRefType::get({rows, kLane}, sourceType.getElementType());
  Value pack = b.create<memref::AllocOp>(loc, packType);
  Value zeroValue = b.create<arith::ConstantIntOp>(loc, 0, 8);
  b.create<linalg::FillOp>(loc, zeroValue, pack);

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
  Value kUpper = b.create<arith::ConstantIndexOp>(loc, kTiles);
  Value kEnd = b.create<arith::ConstantIndexOp>(loc, k);
  Value nBase = b.create<arith::ConstantIndexOp>(loc, n0);
  Value nEnd = b.create<arith::ConstantIndexOp>(loc, n0 + nChunk);

  auto kLoop = b.create<scf::ForOp>(loc, zero, kUpper, one);
  b.setInsertionPointToStart(kLoop.getBody());
  Value kt = kLoop.getInductionVar();
  auto rLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(rLoop.getBody());
  Value r = rLoop.getInductionVar();
  auto cLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(cLoop.getBody());
  Value c = cLoop.getInductionVar();

  Value srcRow = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, kt, sixteen), r);
  Value srcCol = b.create<arith::AddIOp>(loc, nBase, c);
  Value dstRow = srcRow;
  Value rowValid =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, srcRow, kEnd);
  Value colValid =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, srcCol, nEnd);
  Value valid = b.create<arith::AndIOp>(loc, rowValid, colValid);
  auto load = b.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
  b.setInsertionPointToStart(&load.getThenRegion().front());
  Value value =
      b.create<memref::LoadOp>(loc, source, ValueRange{srcRow, srcCol});
  b.create<memref::StoreOp>(loc, value, pack, ValueRange{dstRow, c});
  b.setInsertionPointAfter(kLoop);
  return pack;
}

static void unpackOutput(OpBuilder &b, Location loc, Value source, Value target,
                         int64_t m0, int64_t mChunk, int64_t n0,
                         int64_t nChunk) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value mUpper = b.create<arith::ConstantIndexOp>(loc, mChunk);
  Value nUpper = b.create<arith::ConstantIndexOp>(loc, nChunk);
  Value mBase = b.create<arith::ConstantIndexOp>(loc, m0);
  Value nBase = b.create<arith::ConstantIndexOp>(loc, n0);
  auto mLoop = b.create<scf::ForOp>(loc, zero, mUpper, one);
  b.setInsertionPointToStart(mLoop.getBody());
  Value m = mLoop.getInductionVar();
  auto nLoop = b.create<scf::ForOp>(loc, zero, nUpper, one);
  b.setInsertionPointToStart(nLoop.getBody());
  Value n = nLoop.getInductionVar();
  Value value = b.create<memref::LoadOp>(loc, source, ValueRange{m, n});
  Value dstRow = b.create<arith::AddIOp>(loc, mBase, m);
  Value dstCol = b.create<arith::AddIOp>(loc, nBase, n);
  b.create<memref::StoreOp>(loc, value, target, ValueRange{dstRow, dstCol});
  b.setInsertionPointAfter(mLoop);
}

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
    if (M <= 0 || K <= 0 || N <= 0)
      return op.emitError("M/N/K must be positive");
    int64_t requiredDwBytes = perChannel ? N * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");
    const int64_t kTiles = ceilDiv(K, kLane);
    const int64_t aRowsPerMTile = kTiles * kLane;
    const int64_t maxMChunk = (kBankDepth / aRowsPerMTile) * kLane;
    if (maxMChunk == 0)
      return op.emitError("K exceeds activation-bank packing capacity");

    SmallVector<Value> activationPacks;
    SmallVector<Value> weightPacks;
    SmallVector<OutputPack> outputPacks;
    Value aF = allocBank(b, loc, 1, 4);
    Value aI = allocBank(b, loc, 1, 1);
    Value bI = allocBank(b, loc, 1, 1);
    Value cB = allocBank(b, loc, 1, 4);
    Value cF = allocBank(b, loc, 1, 4);

    // Quantize each activation chunk once, then reuse its Da and INT8 bank
    // across every output panel. The panel loop only changes the weight/Dw
    // slice and the destination tile.
    for (int64_t m0 = 0; m0 < M;) {
      int64_t mChunk = std::min(maxMChunk, M - m0);
      if (mChunk > 4095)
        return op.emitError("matmul M chunk exceeds matrix cfg field");

      const int64_t aRows = ceilDiv(mChunk, kLane) * aRowsPerMTile;
      Value daAddr = createI64Const(b, loc, 0);
      Value aPack = packActivation(b, loc, op.getAMemArray(), m0, mChunk, K);
      activationPacks.push_back(aPack);
      Value aL = mvinBank(b, loc, aPack, aF, aRows, 1);
      Value aQ = b.create<BankFp2IntOp>(loc, aI.getType(), aL, aI,
                                        createI64Const(b, loc, aRows), daAddr);

      for (int64_t n0 = 0; n0 < N; n0 += kLane) {
        int64_t nChunk = std::min(kLane, N - n0);
        int64_t dwAddr = dwAddrBase + (perChannel ? n0 * 4 : 0);
        Value dwAddrValue = createI64Const(b, loc, dwAddr);
        uint64_t cfg =
            matrixRs2((uint64_t)mChunk, (uint64_t)nChunk, (uint64_t)K);
        Value bPack = packWeight(b, loc, op.getBMemArray(), n0, K, nChunk);
        weightPacks.push_back(bPack);
        Value bL = mvinBank(b, loc, bPack, bI, aRowsPerMTile);
        Value cO =
            createBankSMatMul(b, loc, cB.getType(), aQ, bL, cB,
                              createI64Const(b, loc, (int64_t)cfg), mChunk);
        Value fp =
            perChannel
                ? b.create<BankInt2FpChannelOp>(loc, cF.getType(), cO, cF,
                                                createI64Const(b, loc, mChunk),
                                                daAddr, dwAddrValue)
                      .getResult()
                : b.create<BankInt2FpTensorOp>(loc, cF.getType(), cO, cF,
                                               createI64Const(b, loc, mChunk),
                                               daAddr, dwAddrValue)
                      .getResult();
        auto cPackType = MemRefType::get({mChunk, kLane}, b.getF32Type());
        Value cPack = b.create<memref::AllocOp>(loc, cPackType);
        mvoutBank(b, loc, cPack, fp, mChunk, 1);
        outputPacks.push_back({cPack, m0, mChunk, n0, nChunk});
      }
      m0 += mChunk;
    }

    releaseBank(b, loc, aF);
    releaseBank(b, loc, aI);
    releaseBank(b, loc, bI);
    releaseBank(b, loc, cB);
    releaseBank(b, loc, cF);

    b.create<FenceOp>(loc);
    for (const OutputPack &pack : outputPacks)
      unpackOutput(b, loc, pack.value, op.getCMemArray(), pack.m0, pack.mChunk,
                   pack.n0, pack.nChunk);
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

void mlir::buddy::populatePebbleSMatMulMatmulToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatrixMatmulToBankSSAPattern>(patterns.getContext());
}

void mlir::buddy::populatePebbleLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  populatePebbleSMatMulMatmulToBankSSAPatterns(patterns);
  populatePebbleIm2colMatmulToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
}
