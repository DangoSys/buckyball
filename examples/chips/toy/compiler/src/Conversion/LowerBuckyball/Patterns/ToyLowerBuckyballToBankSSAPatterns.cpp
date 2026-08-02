//===- ToyLowerBuckyballToBankSSAPatterns.cpp - Toy bank-SSA patterns -----===//

#include "Conversion/LowerBuckyball/Patterns/ToyLowerBuckyballPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

static Value cstI64(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                           builder.getI64IntegerAttr(value));
}

static Value cstF32(OpBuilder &builder, Location loc, float value) {
  return builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                           builder.getF32FloatAttr(value));
}

static Value packF32BitsAsI64(OpBuilder &builder, Location loc, Value f32Val) {
  Value i32Bits =
      builder.create<arith::BitcastOp>(loc, builder.getI32Type(), f32Val);
  return builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), i32Bits);
}

static Value buildTileAbsMax(PatternRewriter &rewriter, Location loc, Value mem,
                             uint64_t rows, uint64_t cols) {
  auto maxTy = MemRefType::get({1}, rewriter.getF32Type());
  Value maxBuf = rewriter.create<memref::AllocOp>(loc, maxTy);

  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsIdx = rewriter.create<arith::ConstantIndexOp>(loc, rows);
  Value colsIdx = rewriter.create<arith::ConstantIndexOp>(loc, cols);
  Value zeroF32 = cstF32(rewriter, loc, 0.0f);
  rewriter.create<memref::StoreOp>(loc, zeroF32, maxBuf, ValueRange{zeroIdx});

  auto rowLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, rowsIdx, oneIdx);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto colLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, colsIdx, oneIdx);
  rewriter.setInsertionPointToStart(colLoop.getBody());

  Value elem = rewriter.create<memref::LoadOp>(
      loc, mem,
      ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
  if (elem.getType() != rewriter.getF32Type())
    elem = rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), elem);
  Value neg = rewriter.create<arith::NegFOp>(loc, elem);
  Value abs = rewriter.create<arith::MaximumFOp>(loc, elem, neg);
  Value cur = rewriter.create<memref::LoadOp>(loc, maxBuf, ValueRange{zeroIdx});
  Value upd = rewriter.create<arith::MaximumFOp>(loc, cur, abs);
  rewriter.create<memref::StoreOp>(loc, upd, maxBuf, ValueRange{zeroIdx});

  rewriter.setInsertionPointAfter(rowLoop);
  Value result =
      rewriter.create<memref::LoadOp>(loc, maxBuf, ValueRange{zeroIdx});
  rewriter.create<memref::DeallocOp>(loc, maxBuf);
  return result;
}

static Value buildQuantScale(PatternRewriter &rewriter, Location loc,
                             Value maxAbs) {
  Value zeroF32 = cstF32(rewriter, loc, 0.0f);
  Value oneF32 = cstF32(rewriter, loc, 1.0f);
  Value qmaxF32 = cstF32(rewriter, loc, 127.0f);
  Value hasData = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                 maxAbs, zeroF32);
  Value scaled = rewriter.create<arith::DivFOp>(loc, qmaxF32, maxAbs);
  return rewriter.create<arith::SelectOp>(loc, hasData, scaled, oneF32);
}

static LogicalResult getStaticRowStrideDiv16(MemRefType type, uint64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % 16 != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = static_cast<uint64_t>(strides[0] / 16);
  return success();
}

class MatrixMatmulToBankSSAPattern : public OpRewritePattern<MatrixMatmulOp> {
public:
  using OpRewritePattern<MatrixMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatrixMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMem = op.getAMemArray();
    Value bMem = op.getBMemArray();
    Value cMem = op.getCMemArray();

    auto aTy = dyn_cast<MemRefType>(aMem.getType());
    auto bTy = dyn_cast<MemRefType>(bMem.getType());
    auto cTy = dyn_cast<MemRefType>(cMem.getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() ||
        !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "bb_matmul requires static rank-2 memrefs");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb)
      return rewriter.notifyMatchFailure(op, "inner dimensions must match");
    if (cTy.getShape()[0] != static_cast<int64_t>(m) ||
        cTy.getShape()[1] != static_cast<int64_t>(n))
      return rewriter.notifyMatchFailure(op, "output dimensions must match");
    if (m % 16 != 0 || k % 16 != 0 || n % 16 != 0)
      return rewriter.notifyMatchFailure(
          op,
          "buckyball.matrix_matmul requires M, K and N to be multiples of 16");

    uint64_t strideB = 0;
    uint64_t strideC = 0;
    if (failed(getStaticRowStrideDiv16(bTy, strideB)))
      return rewriter.notifyMatchFailure(
          op, "B requires static strided<[row,1]> and row % 16 == 0");
    if (failed(getStaticRowStrideDiv16(cTy, strideC)))
      return rewriter.notifyMatchFailure(
          op, "C requires static strided<[row,1]> and row % 16 == 0");

    constexpr uint64_t tile = 16;
    uint64_t depthC = tile;

    OpBuilder::InsertionGuard guard(rewriter);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mUpper = rewriter.create<arith::ConstantIndexOp>(loc, m);
    Value nUpper = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value kUpper = rewriter.create<arith::ConstantIndexOp>(loc, k);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, tile);

    auto mLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, mUpper, step);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mIv = mLoop.getInductionVar();

    auto nLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, nUpper, step);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nIv = nLoop.getInductionVar();

    Value aTile = rewriter.create<memref::SubViewOp>(
        loc, aMem, SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(tile),
                                  rewriter.getIndexAttr(k)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});
    Value bTile = rewriter.create<memref::SubViewOp>(
        loc, bMem, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), nIv},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(k),
                                  rewriter.getIndexAttr(tile)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});
    Value cTile = rewriter.create<memref::SubViewOp>(
        loc, cMem, SmallVector<OpFoldResult>{mIv, nIv},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(tile),
                                  rewriter.getIndexAttr(tile)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(1)});

    Value maxA = buildTileAbsMax(rewriter, loc, aTile, tile, k);
    Value maxB = buildTileAbsMax(rewriter, loc, bTile, k, tile);
    Value scaleAF32 = buildQuantScale(rewriter, loc, maxA);
    Value scaleBF32 = buildQuantScale(rewriter, loc, maxB);
    Value scaleABits = packF32BitsAsI64(rewriter, loc, scaleAF32);
    Value scaleBBits = packF32BitsAsI64(rewriter, loc, scaleBF32);
    Value oneF32 = cstF32(rewriter, loc, 1.0f);
    Value scaleProd = rewriter.create<arith::MulFOp>(loc, scaleAF32, scaleBF32);
    Value dequantScaleF32 =
        rewriter.create<arith::DivFOp>(loc, oneF32, scaleProd);
    Value dequantScaleBits = packF32BitsAsI64(rewriter, loc, dequantScaleF32);

    auto cI32 = rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());
    cI32->setAttr("col", rewriter.getI64IntegerAttr(4));
    auto cFp32 = rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());
    cFp32->setAttr("col", rewriter.getI64IntegerAttr(4));

    auto zeroI32Ty =
        MemRefType::get({(int64_t)tile, (int64_t)tile}, rewriter.getI32Type());
    Value zeroI32Buf = rewriter.create<memref::AllocOp>(loc, zeroI32Ty);
    Value zeroI32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value tileIdx = rewriter.create<arith::ConstantIndexOp>(loc, tile);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto zRow = rewriter.create<scf::ForOp>(loc, zeroIdx, tileIdx, oneIdx);
    rewriter.setInsertionPointToStart(zRow.getBody());
    auto zCol = rewriter.create<scf::ForOp>(loc, zeroIdx, tileIdx, oneIdx);
    rewriter.setInsertionPointToStart(zCol.getBody());
    rewriter.create<memref::StoreOp>(
        loc, zeroI32, zeroI32Buf,
        ValueRange{zRow.getInductionVar(), zCol.getInductionVar()});
    rewriter.setInsertionPointAfter(zRow);

    auto cZero = rewriter.create<BankMvinOp>(
        loc, rewriter.getI64Type(), zeroI32Buf, cI32.getBank(),
        cstI64(rewriter, loc, depthC), cstI64(rewriter, loc, 1));
    rewriter.create<memref::DeallocOp>(loc, zeroI32Buf);

    SmallVector<Value> kIterArgs = {cZero.getBankOut()};
    auto kLoop = rewriter.create<scf::ForOp>(
        loc, zeroIdx, kUpper, step, kIterArgs,
        [&](OpBuilder &b, Location bodyLoc, Value kIv, ValueRange args) {
          Value cIn = args[0];

          Value aKTile = b.create<memref::SubViewOp>(
              bodyLoc, aMem, SmallVector<OpFoldResult>{mIv, kIv},
              SmallVector<OpFoldResult>{b.getIndexAttr(tile),
                                        b.getIndexAttr(tile)},
              SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          Value bKTile = b.create<memref::SubViewOp>(
              bodyLoc, bMem, SmallVector<OpFoldResult>{kIv, nIv},
              SmallVector<OpFoldResult>{b.getIndexAttr(tile),
                                        b.getIndexAttr(tile)},
              SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

          uint64_t strideAK = k / tile;

          auto aFp32 = b.create<BankAllocOp>(bodyLoc, b.getI64Type());
          aFp32->setAttr("col", b.getI64IntegerAttr(4));
          auto bFp32 = b.create<BankAllocOp>(bodyLoc, b.getI64Type());
          bFp32->setAttr("col", b.getI64IntegerAttr(4));
          auto aI8 = b.create<BankAllocOp>(bodyLoc, b.getI64Type());
          auto bI8 = b.create<BankAllocOp>(bodyLoc, b.getI64Type());
          auto aI8T = b.create<BankAllocOp>(bodyLoc, b.getI64Type());

          auto aLoad = b.create<BankMvinOp>(
              bodyLoc, b.getI64Type(), aKTile, aFp32.getBank(),
              cstI64(b, bodyLoc, tile), cstI64(b, bodyLoc, strideAK));
          auto bLoad = b.create<BankMvinOp>(
              bodyLoc, b.getI64Type(), bKTile, bFp32.getBank(),
              cstI64(b, bodyLoc, tile), cstI64(b, bodyLoc, strideB));

          auto aQuant = b.create<BankFp2IntOp>(
              bodyLoc, b.getI64Type(), aLoad.getBankOut(), aI8.getBank(),
              cstI64(b, bodyLoc, tile), scaleABits);
          auto bQuant = b.create<BankFp2IntOp>(
              bodyLoc, b.getI64Type(), bLoad.getBankOut(), bI8.getBank(),
              cstI64(b, bodyLoc, tile), scaleBBits);
          b.create<BankReleaseOp>(bodyLoc, aLoad.getBankOut());
          b.create<BankReleaseOp>(bodyLoc, bLoad.getBankOut());

          auto aTrans = b.create<BankTransposeOp>(
              bodyLoc, b.getI64Type(), aQuant.getOutBankOut(), aI8T.getBank(),
              cstI64(b, bodyLoc, tile), cstI64(b, bodyLoc, 8));
          b.create<BankReleaseOp>(bodyLoc, aQuant.getOutBankOut());

          auto cMul = b.create<BankMulWarp16Op>(
              bodyLoc, b.getI64Type(), aTrans.getOutBankOut(),
              bQuant.getOutBankOut(), cIn, cstI64(b, bodyLoc, tile),
              cstI64(b, bodyLoc, 0));
          b.create<BankReleaseOp>(bodyLoc, aTrans.getOutBankOut());
          b.create<BankReleaseOp>(bodyLoc, bQuant.getOutBankOut());
          b.create<scf::YieldOp>(bodyLoc, ValueRange{cMul.getWrBankOut()});
        });

    Value cAcc = kLoop.getResult(0);
    auto cDequant = rewriter.create<BankInt2FpOp>(
        loc, rewriter.getI64Type(), cAcc, cFp32.getBank(),
        cstI64(rewriter, loc, depthC), dequantScaleBits);
    rewriter.create<BankReleaseOp>(loc, cAcc);

    auto cStore = rewriter.create<BankMvoutOp>(
        loc, rewriter.getI64Type(), cTile, cDequant.getOutBankOut(),
        cstI64(rewriter, loc, depthC), cstI64(rewriter, loc, strideC));
    rewriter.create<FenceOp>(loc);
    rewriter.create<BankReleaseOp>(loc, cStore.getBankOut());

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populateToyLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatrixMatmulToBankSSAPattern>(patterns.getContext());
}
