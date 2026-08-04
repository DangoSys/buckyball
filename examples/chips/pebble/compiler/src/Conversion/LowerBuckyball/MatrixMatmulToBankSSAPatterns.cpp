//===- MatrixMatmulToBankSSAPatterns.cpp - f32 matmul -> BankMatrix -------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Utils/BankUtils.h"
#include "Utils/QuantUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kTile = 16;

static LogicalResult rowStrideDiv16(MemRefType ty, int64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % kTile != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = strides[0] / kTile;
  return success();
}

static void addI32Tile(OpBuilder &b, Location loc, Value dst, Value src) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value n = b.create<arith::ConstantIndexOp>(loc, kTile);
  auto r = b.create<scf::ForOp>(loc, zero, n, one);
  b.setInsertionPointToStart(r.getBody());
  auto c = b.create<scf::ForOp>(loc, zero, n, one);
  b.setInsertionPointToStart(c.getBody());
  Value a = b.create<memref::LoadOp>(
      loc, dst, ValueRange{r.getInductionVar(), c.getInductionVar()});
  Value t = b.create<memref::LoadOp>(
      loc, src, ValueRange{r.getInductionVar(), c.getInductionVar()});
  b.create<memref::StoreOp>(
      loc, b.create<arith::AddIOp>(loc, a, t), dst,
      ValueRange{r.getInductionVar(), c.getInductionVar()});
  b.setInsertionPointAfter(r);
}

class MatrixMatmulToBankSSAPattern : public OpRewritePattern<MatrixMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatrixMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto aTy = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto bTy = dyn_cast<MemRefType>(op.getBMemArray().getType());
    auto cTy = dyn_cast<MemRefType>(op.getCMemArray().getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() ||
        !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!aTy.getElementType().isF32() || !bTy.getElementType().isF32() ||
        !cTy.getElementType().isF32())
      return b.notifyMatchFailure(op, "pebble bank-ssa path is f32 only");

    int64_t M = aTy.getShape()[0], K = aTy.getShape()[1];
    int64_t Kb = bTy.getShape()[0], N = bTy.getShape()[1];
    if (K != Kb || cTy.getShape()[0] != M || cTy.getShape()[1] != N)
      return op.emitError("matmul shape mismatch");
    if (M % kTile || K % kTile || N % kTile)
      return op.emitError("M/N/K must be multiples of 16");

    int64_t strideA = 0, strideB = 0, strideC = 0;
    if (failed(rowStrideDiv16(aTy, strideA)) ||
        failed(rowStrideDiv16(bTy, strideB)) ||
        failed(rowStrideDiv16(cTy, strideC)))
      return op.emitError("A/B/C need static strided<[row,1]> with row%16==0");

    uint64_t cfg = packBits(kTile, 0, 11) | packBits(kTile, 12, 23) |
                   packBits(kTile, 24, 35);
    Value cfgV = createI64Const(b, loc, (int64_t)cfg);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value step = b.create<arith::ConstantIndexOp>(loc, kTile);
    Value mU = b.create<arith::ConstantIndexOp>(loc, M);
    Value nU = b.create<arith::ConstantIndexOp>(loc, N);
    Value kU = b.create<arith::ConstantIndexOp>(loc, K);
    Value i0 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                           b.getI32IntegerAttr(0));

    auto mL = b.create<scf::ForOp>(loc, zero, mU, step);
    b.setInsertionPointToStart(mL.getBody());
    Value mIv = mL.getInductionVar();
    auto nL = b.create<scf::ForOp>(loc, zero, nU, step);
    b.setInsertionPointToStart(nL.getBody());
    Value nIv = nL.getInductionVar();

    Value aStrip = b.create<memref::SubViewOp>(
        loc, op.getAMemArray(),
        SmallVector<OpFoldResult>{mIv, b.getIndexAttr(0)},
        SmallVector<OpFoldResult>{b.getIndexAttr(kTile), b.getIndexAttr(K)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value bStrip = b.create<memref::SubViewOp>(
        loc, op.getBMemArray(),
        SmallVector<OpFoldResult>{b.getIndexAttr(0), nIv},
        SmallVector<OpFoldResult>{b.getIndexAttr(K), b.getIndexAttr(kTile)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value scaleAF = quantScale(b, loc, absMaxF32(b, loc, aStrip, kTile, K));
    Value scaleBF = quantScale(b, loc, absMaxF32(b, loc, bStrip, K, kTile));
    Value scaleA = packF32BitsAsI64(b, loc, scaleAF);
    Value scaleB = packF32BitsAsI64(b, loc, scaleBF);
    Value scaleD =
        packF32BitsAsI64(b, loc, dequantScale(b, loc, scaleAF, scaleBF));

    auto i32Tile = MemRefType::get({kTile, kTile}, b.getI32Type());
    Value acc = b.create<memref::AllocOp>(loc, i32Tile);
    Value tmp = b.create<memref::AllocOp>(loc, i32Tile);
    b.create<linalg::FillOp>(loc, i0, acc);

    auto kL = b.create<scf::ForOp>(loc, zero, kU, step);
    b.setInsertionPointToStart(kL.getBody());
    Value kIv = kL.getInductionVar();

    Value aTile = b.create<memref::SubViewOp>(
        loc, op.getAMemArray(), SmallVector<OpFoldResult>{mIv, kIv},
        SmallVector<OpFoldResult>{b.getIndexAttr(kTile), b.getIndexAttr(kTile)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value bTile = b.create<memref::SubViewOp>(
        loc, op.getBMemArray(), SmallVector<OpFoldResult>{kIv, nIv},
        SmallVector<OpFoldResult>{b.getIndexAttr(kTile), b.getIndexAttr(kTile)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

    Value aF = allocBank(b, loc, 1, 4);
    Value aI = allocBank(b, loc, 1, 1);
    Value aL = mvinBank(b, loc, aTile, aF, kTile, strideA);
    Value aQ = b.create<BankFp2IntOp>(loc, aI.getType(), aL, aI,
                                      createI64Const(b, loc, kTile), scaleA);
    releaseBank(b, loc, aL);

    Value bF = allocBank(b, loc, 1, 4);
    Value bI = allocBank(b, loc, 1, 1);
    Value bL = mvinBank(b, loc, bTile, bF, kTile, strideB);
    Value bQ = b.create<BankFp2IntOp>(loc, bI.getType(), bL, bI,
                                      createI64Const(b, loc, kTile), scaleB);
    releaseBank(b, loc, bL);

    Value cB = allocBank(b, loc, 1, 4);
    Value cO = b.create<BankMatrixOp>(loc, cB.getType(), aQ, bQ, cB, cfgV);
    releaseBank(b, loc, aQ);
    releaseBank(b, loc, bQ);
    mvoutBank(b, loc, tmp, cO, kTile);
    releaseBank(b, loc, cO);
    addI32Tile(b, loc, acc, tmp);
    b.setInsertionPointAfter(kL);

    Value cTile = b.create<memref::SubViewOp>(
        loc, op.getCMemArray(), SmallVector<OpFoldResult>{mIv, nIv},
        SmallVector<OpFoldResult>{b.getIndexAttr(kTile), b.getIndexAttr(kTile)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value accF = allocBank(b, loc, 1, 4);
    Value accL = mvinBank(b, loc, acc, accF, kTile);
    Value fp = b.create<BankInt2FpOp>(loc, accL.getType(), accL, accL,
                                      createI64Const(b, loc, kTile), scaleD);
    mvoutBank(b, loc, cTile, fp, kTile, strideC);
    releaseBank(b, loc, fp);
    b.create<memref::DeallocOp>(loc, acc);
    b.create<memref::DeallocOp>(loc, tmp);
    b.setInsertionPointAfter(mL);

    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populatePebbleLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatrixMatmulToBankSSAPattern>(patterns.getContext());
}
