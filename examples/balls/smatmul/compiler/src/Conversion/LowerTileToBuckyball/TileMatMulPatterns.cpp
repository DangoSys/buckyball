//===- TileMatMulPatterns.cpp - tile.matmul -> buckyball.smatmul_matmul
//----===//
//
// Non-quantized matmul is split into bank-sized panels. Quantized matmul
// retains its full N dimension so bank lowering can quantize one activation
// chunk and hand it to every output panel.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerTileToBuckyball/LowerTileToBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Tile/TileOps.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::tile;
using mlir::buddy::aMvinDepthLines;
using mlir::buddy::bMvinDepthLines;
using mlir::buddy::ceilDiv;
using mlir::buddy::cMvoutDepthLines;
using mlir::buddy::kBankLane;

namespace {

class TileMatMulLowering : public OpRewritePattern<TileMatMulOp> {
public:
  explicit TileMatMulLowering(MLIRContext *context, int64_t /*bankWidthBytes*/,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(TileMatMulOp tileMatMulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileMatMulOp.getLoc();

    Value aMemArray = tileMatMulOp.getAMemArray();
    Value bMemArray = tileMatMulOp.getBMemArray();
    Value cMemArray = tileMatMulOp.getCMemArray();

    auto aType = cast<MemRefType>(aMemArray.getType());
    auto bType = cast<MemRefType>(bMemArray.getType());
    auto cType = cast<MemRefType>(cMemArray.getType());

    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    auto cShape = cType.getShape();
    size_t M = aShape[aShape.size() - 2];
    size_t K = aShape[aShape.size() - 1];
    size_t N = bShape[bShape.size() - 1];

    if (bShape[bShape.size() - 2] != (int64_t)K ||
        cShape[cShape.size() - 2] != (int64_t)M ||
        cShape[cShape.size() - 1] != (int64_t)N)
      return tileMatMulOp.emitError("matmul input/output shapes mismatch");

    const bool isQuantized = aType.getElementType().isF32() &&
                             bType.getElementType().isInteger(8) &&
                             cType.getElementType().isF32();
    const bool isWideFloat =
        (aType.getElementType().isF32() || aType.getElementType().isBF16()) &&
        bType.getElementType() == aType.getElementType() &&
        cType.getElementType() == aType.getElementType();
    const bool isInt8 = aType.getElementType().isInteger(8) &&
                        bType.getElementType().isInteger(8) &&
                        cType.getElementType().isInteger(32);
    if (!isQuantized && !isWideFloat && !isInt8)
      return tileMatMulOp.emitError(
          "requires matching floating-point types, FP32 x INT8 -> FP32, "
          "or INT8 x INT8 -> INT32");
    auto dwAddrAttr = tileMatMulOp->getAttrOfType<IntegerAttr>("dw_addr");
    auto dwBytesAttr = tileMatMulOp->getAttrOfType<IntegerAttr>("dw_bytes");
    auto perChannelAttr = tileMatMulOp->getAttrOfType<BoolAttr>("per_channel");
    if (isQuantized && (!dwAddrAttr || !dwBytesAttr || !perChannelAttr ||
                        dwAddrAttr.getInt() < 16))
      return tileMatMulOp.emitError(
          "quantized matmul requires RAX Dw metadata");

    size_t M_pad = ceilDiv(M, kBankLane) * kBankLane;
    size_t K_pad = ceilDiv(K, kBankLane) * kBankLane;
    size_t N_pad = ceilDiv(N, kBankLane) * kBankLane;
    bool needPadding = (M_pad != M) || (K_pad != K) || (N_pad != N);

    Value aMemArrayPadded = aMemArray;
    Value bMemArrayPadded = bMemArray;
    Value cMemArrayPadded = cMemArray;

    if (needPadding) {
      auto aPadType = MemRefType::get({(int64_t)M_pad, (int64_t)K_pad},
                                      aType.getElementType());
      auto bPadType = MemRefType::get({(int64_t)K_pad, (int64_t)N_pad},
                                      bType.getElementType());
      auto cPadType = MemRefType::get({(int64_t)M_pad, (int64_t)N_pad},
                                      cType.getElementType());

      aMemArrayPadded = rewriter.create<memref::AllocOp>(loc, aPadType);
      bMemArrayPadded = rewriter.create<memref::AllocOp>(loc, bPadType);
      cMemArrayPadded = rewriter.create<memref::AllocOp>(loc, cPadType);

      Value aZero = rewriter.create<arith::ConstantOp>(
          loc, aType.getElementType(),
          rewriter.getZeroAttr(aType.getElementType()));
      Value bZero = rewriter.create<arith::ConstantOp>(
          loc, bType.getElementType(),
          rewriter.getZeroAttr(bType.getElementType()));
      Value cZero = rewriter.create<arith::ConstantOp>(
          loc, cType.getElementType(),
          rewriter.getZeroAttr(cType.getElementType()));
      rewriter.create<linalg::FillOp>(loc, aZero, aMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, bZero, bMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, cZero, cMemArrayPadded);

      Value aView = rewriter.create<memref::SubViewOp>(
          loc, aMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(M),
                                    rewriter.getIndexAttr(K)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, aMemArray, aView);

      Value bView = rewriter.create<memref::SubViewOp>(
          loc, bMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(K),
                                    rewriter.getIndexAttr(N)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, bMemArray, bView);
    }

    size_t M_tiling = needPadding ? M_pad : M;
    size_t K_tiling = needPadding ? K_pad : K;
    size_t N_tiling = needPadding ? N_pad : N;

    if (bankDepth <= 0 || (size_t)bankDepth % kBankLane != 0)
      return tileMatMulOp.emitError(
          "bankDepth must be a positive multiple of bank lane");
    if (M_tiling % kBankLane || K_tiling % kBankLane || N_tiling % kBankLane)
      return tileMatMulOp.emitError("M/N/K must be multiples of bank lane");
    if (isQuantized) {
      const int64_t dwRequired = perChannelAttr.getValue() ? N_tiling * 4 : 4;
      if (dwBytesAttr.getInt() < dwRequired)
        return tileMatMulOp.emitError("Dw scale image does not cover padded N");
    }

    // Bank unit on M (depth) and N (lane). Quantized matmul carries one
    // logical K into SMatMul so FP2INT computes one activation scale.
    const size_t mTileSize = std::min((size_t)bankDepth, M_tiling);
    const size_t nTileSize = kBankLane;
    const size_t kTileSize =
        (isWideFloat || isQuantized) ? K_tiling : kBankLane;
    if (M_tiling % mTileSize != 0)
      return tileMatMulOp.emitError("M does not split into bank-depth units");

    if (!isWideFloat && !isQuantized) {
      if (aMvinDepthLines(mTileSize, kTileSize) > (size_t)bankDepth ||
          bMvinDepthLines(kTileSize, nTileSize) > (size_t)bankDepth ||
          cMvoutDepthLines(mTileSize, nTileSize) > (size_t)bankDepth)
        return tileMatMulOp.emitError(
            "bank-unit matmul exceeds 8-bit mvin or 32-bit mvout depth");
    }

    const size_t kTileNum = ceilDiv(K_tiling, kTileSize);

    OpBuilder::InsertionGuard guard(rewriter);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mStepVal = rewriter.create<arith::ConstantIndexOp>(loc, mTileSize);
    Value kStepVal = rewriter.create<arith::ConstantIndexOp>(loc, kTileSize);
    Value mUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, M_tiling);
    Value kUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, K_tiling);

    auto mLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, mUpperVal, mStepVal);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mIv = mLoop.getInductionVar();

    if (isQuantized) {
      Value aTile = rewriter.create<memref::SubViewOp>(
          loc, aMemArrayPadded,
          SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(K_tiling)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value bTile = rewriter.create<memref::SubViewOp>(
          loc, bMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(K_tiling),
                                    rewriter.getIndexAttr(N_tiling)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value cTile = rewriter.create<memref::SubViewOp>(
          loc, cMemArrayPadded,
          SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(N_tiling)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, cType.getElementType(),
          rewriter.getZeroAttr(cType.getElementType()));
      rewriter.create<linalg::FillOp>(loc, zero, cTile);
      auto smatmul = rewriter.create<SMatMulMatmulOp>(loc, aTile, bTile, cTile);
      smatmul->setAttr("dwAddr", dwAddrAttr);
      smatmul->setAttr("dwBytes", dwBytesAttr);
      smatmul->setAttr("perChannel", perChannelAttr);
    } else {
      // N is static. Non-quantized matmul remains one physical N panel per op.
      for (size_t n0 = 0; n0 < N_tiling; n0 += nTileSize) {
        Value cTile = rewriter.create<memref::SubViewOp>(
            loc, cMemArrayPadded,
            SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(n0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                      rewriter.getIndexAttr(nTileSize)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        auto cElemType = cType.getElementType();
        Value zero = rewriter.create<arith::ConstantOp>(
            loc, cElemType, rewriter.getZeroAttr(cElemType));
        rewriter.create<linalg::FillOp>(loc, zero, cTile);

        if (kTileNum == 1) {
          Value aTile = rewriter.create<memref::SubViewOp>(
              loc, aMemArrayPadded,
              SmallVector<OpFoldResult>{mIv, rewriter.getIndexAttr(0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                        rewriter.getIndexAttr(kTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          Value bTile = rewriter.create<memref::SubViewOp>(
              loc, bMemArrayPadded,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(n0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(kTileSize),
                                        rewriter.getIndexAttr(nTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          auto smatmul =
              rewriter.create<SMatMulMatmulOp>(loc, aTile, bTile, cTile);
        } else {
          auto kLoop =
              rewriter.create<scf::ForOp>(loc, zeroIdx, kUpperVal, kStepVal);
          rewriter.setInsertionPointToStart(kLoop.getBody());
          Value kIv = kLoop.getInductionVar();
          Value aTile = rewriter.create<memref::SubViewOp>(
              loc, aMemArrayPadded, SmallVector<OpFoldResult>{mIv, kIv},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                        rewriter.getIndexAttr(kTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          Value bTile = rewriter.create<memref::SubViewOp>(
              loc, bMemArrayPadded,
              SmallVector<OpFoldResult>{kIv, rewriter.getIndexAttr(n0)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(kTileSize),
                                        rewriter.getIndexAttr(nTileSize)},
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)});
          if (isWideFloat)
            return tileMatMulOp.emitError(
                "FP32 matmul must pass full K in one smatmul_matmul");
          // 8-bit legalize overwrites C; accumulate via host partial.
          auto partialTy = MemRefType::get(
              {(int64_t)mTileSize, (int64_t)nTileSize}, cElemType);
          Value partial = rewriter.create<memref::AllocOp>(loc, partialTy);
          rewriter.create<linalg::FillOp>(loc, zero, partial);
          rewriter.create<SMatMulMatmulOp>(loc, aTile, bTile, partial);
          rewriter.create<linalg::AddOp>(loc, ValueRange{cTile, partial},
                                         ValueRange{cTile});
          rewriter.create<memref::DeallocOp>(loc, partial);
          rewriter.setInsertionPointAfter(kLoop);
        }
      }
    }

    rewriter.setInsertionPointAfter(mLoop);

    if (needPadding) {
      Value cView = rewriter.create<memref::SubViewOp>(
          loc, cMemArrayPadded,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(M),
                                    rewriter.getIndexAttr(N)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, cView, cMemArray);
      rewriter.create<memref::DeallocOp>(loc, aMemArrayPadded);
      rewriter.create<memref::DeallocOp>(loc, bMemArrayPadded);
      rewriter.create<memref::DeallocOp>(loc, cMemArrayPadded);
    }

    rewriter.eraseOp(tileMatMulOp);
    return success();
  }

private:
  int64_t bankDepth;
};

} // namespace

namespace mlir::buddy {
void populateSMatMulBallTileLoweringPatterns(RewritePatternSet &patterns,
                                             int64_t bankWidthBytes,
                                             int64_t bankDepth,
                                             int64_t bankNum) {
  patterns.add<TileMatMulLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
}
} // namespace mlir::buddy
