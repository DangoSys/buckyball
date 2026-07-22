//===- TileMatMulPatterns.cpp - tile.matmul -> buckyball.matrix_matmul ----===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::tile;
using mlir::buddy::aMvinDepthLines;
using mlir::buddy::bMvinDepthLines;
using mlir::buddy::ceilDiv;
using mlir::buddy::cMvoutDepthLines;
using mlir::buddy::kMatmulTile;
using mlir::buddy::kMaxAccMvoutDepthLines;
using mlir::buddy::kMaxI8MvinDepthLines;

namespace {

class TileMatMulLowering : public OpRewritePattern<TileMatMulOp> {
  size_t computeBankRows(size_t mTileLen, size_t nTileLen,
                         size_t kTileLen) const {
    return mTileLen * kTileLen + kTileLen * nTileLen;
  }

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

    size_t M_pad = ceilDiv(M, 16) * 16;
    size_t K_pad = ceilDiv(K, 16) * 16;
    size_t N_pad = ceilDiv(N, 16) * 16;
    bool needPadding = (M_pad != M) || (K_pad != K) || (N_pad != N);

    Value aMemArrayPadded = aMemArray;
    Value bMemArrayPadded = bMemArray;
    Value cMemArrayPadded = cMemArray;

    if (needPadding) {
      auto elemType = aType.getElementType();

      auto aPadType =
          MemRefType::get({(int64_t)M_pad, (int64_t)K_pad}, elemType);
      auto bPadType =
          MemRefType::get({(int64_t)K_pad, (int64_t)N_pad}, elemType);
      auto cPadType =
          MemRefType::get({(int64_t)M_pad, (int64_t)N_pad}, elemType);

      aMemArrayPadded = rewriter.create<memref::AllocOp>(loc, aPadType);
      bMemArrayPadded = rewriter.create<memref::AllocOp>(loc, bPadType);
      cMemArrayPadded = rewriter.create<memref::AllocOp>(loc, cPadType);

      Value zero = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getZeroAttr(elemType));
      rewriter.create<linalg::FillOp>(loc, zero, aMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, zero, bMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, zero, cMemArrayPadded);

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

    const size_t mMeta = kMatmulTile;
    const size_t nMeta = kMatmulTile;
    const size_t kMeta = kMatmulTile;

    const size_t mPad = ceilDiv(M_tiling, mMeta) * mMeta;
    const size_t nPad = ceilDiv(N_tiling, nMeta) * nMeta;
    const size_t kPad = ceilDiv(K_tiling, kMeta) * kMeta;

    size_t mTileLen = 1, nTileLen = 1, kTileLen = 1;

    for (size_t cand = kTileLen + 1; cand * kMeta <= kPad; ++cand) {
      size_t candSize = cand * kMeta;
      if (computeBankRows(1, 1, cand) > (size_t)bankDepth ||
          aMvinDepthLines(mMeta, candSize) > kMaxI8MvinDepthLines ||
          bMvinDepthLines(candSize, nMeta) > kMaxI8MvinDepthLines)
        break;
      if (kPad % candSize == 0)
        kTileLen = cand;
    }

    const size_t kTileSize = kTileLen * kMeta;

    for (size_t cand = nTileLen + 1; cand * nMeta <= nPad; ++cand) {
      size_t candSize = cand * nMeta;
      if (computeBankRows(1, cand, kTileLen) > (size_t)bankDepth ||
          cMvoutDepthLines(mMeta, candSize) > kMaxAccMvoutDepthLines ||
          bMvinDepthLines(kTileSize, candSize) > kMaxI8MvinDepthLines)
        break;
      if (nPad % candSize == 0)
        nTileLen = cand;
    }

    for (size_t cand = mTileLen + 1; cand * mMeta <= mPad; ++cand) {
      size_t candSize = cand * mMeta;
      if (computeBankRows(cand, nTileLen, kTileLen) > (size_t)bankDepth ||
          cMvoutDepthLines(candSize, nTileLen * nMeta) >
              kMaxAccMvoutDepthLines ||
          aMvinDepthLines(candSize, kTileSize) > kMaxI8MvinDepthLines)
        break;
      if (mPad % candSize == 0)
        mTileLen = cand;
    }

    const size_t mTileSize = mTileLen * mMeta;
    const size_t nTileSize = nTileLen * nMeta;
    const size_t kTileNum = ceilDiv(kPad, kTileSize);

    if (mPad % mTileSize != 0 || nPad % nTileSize != 0 ||
        kPad % kTileSize != 0) {
      return tileMatMulOp.emitError()
             << "padded dims (m=" << mPad << ", n=" << nPad << ", k=" << kPad
             << ") must be multiples of tile sizes (m=" << mTileSize
             << ", n=" << nTileSize << ", k=" << kTileSize
             << "); partial tiles not yet supported";
    }

    OpBuilder::InsertionGuard guard(rewriter);

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mStepVal = rewriter.create<arith::ConstantIndexOp>(loc, mTileSize);
    Value nStepVal = rewriter.create<arith::ConstantIndexOp>(loc, nTileSize);
    Value kStepVal = rewriter.create<arith::ConstantIndexOp>(loc, kTileSize);
    Value mUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, mPad);
    Value nUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, nPad);
    Value kUpperVal = rewriter.create<arith::ConstantIndexOp>(loc, kPad);
    Operation *outerLoop = nullptr;

    if (kTileNum == 1) {
      auto kLoop =
          rewriter.create<scf::ForOp>(loc, zeroIdx, kUpperVal, kStepVal);
      outerLoop = kLoop;
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kIv = kLoop.getInductionVar();

      auto mLoop =
          rewriter.create<scf::ForOp>(loc, zeroIdx, mUpperVal, mStepVal);
      rewriter.setInsertionPointToStart(mLoop.getBody());
      Value mIv = mLoop.getInductionVar();

      auto nLoop =
          rewriter.create<scf::ForOp>(loc, zeroIdx, nUpperVal, nStepVal);
      rewriter.setInsertionPointToStart(nLoop.getBody());
      Value nIv = nLoop.getInductionVar();

      Value aTile = rewriter.create<memref::SubViewOp>(
          loc, aMemArrayPadded, SmallVector<OpFoldResult>{mIv, kIv},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(kTileSize)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value bTile = rewriter.create<memref::SubViewOp>(
          loc, bMemArrayPadded, SmallVector<OpFoldResult>{kIv, nIv},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(kTileSize),
                                    rewriter.getIndexAttr(nTileSize)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      Value cTile = rewriter.create<memref::SubViewOp>(
          loc, cMemArrayPadded, SmallVector<OpFoldResult>{mIv, nIv},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(nTileSize)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});

      rewriter.create<MatrixMatmulOp>(loc, aTile, bTile, cTile);
    } else {
      auto mLoop =
          rewriter.create<scf::ForOp>(loc, zeroIdx, mUpperVal, mStepVal);
      outerLoop = mLoop;
      rewriter.setInsertionPointToStart(mLoop.getBody());
      Value mIv = mLoop.getInductionVar();

      auto nLoop =
          rewriter.create<scf::ForOp>(loc, zeroIdx, nUpperVal, nStepVal);
      rewriter.setInsertionPointToStart(nLoop.getBody());
      Value nIv = nLoop.getInductionVar();

      Value cTile = rewriter.create<memref::SubViewOp>(
          loc, cMemArrayPadded, SmallVector<OpFoldResult>{mIv, nIv},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(mTileSize),
                                    rewriter.getIndexAttr(nTileSize)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});

      auto elemType = aType.getElementType();
      auto partialType =
          MemRefType::get({(int64_t)mTileSize, (int64_t)nTileSize}, elemType);
      Value partial = rewriter.create<memref::AllocOp>(loc, partialType);

      Value zero = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getZeroAttr(elemType));
      rewriter.create<linalg::FillOp>(loc, zero, cTile);

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
          loc, bMemArrayPadded, SmallVector<OpFoldResult>{kIv, nIv},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(kTileSize),
                                    rewriter.getIndexAttr(nTileSize)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});

      rewriter.create<MatrixMatmulOp>(loc, aTile, bTile, partial);

      Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value iUpper = rewriter.create<arith::ConstantIndexOp>(loc, mTileSize);
      Value jUpper = rewriter.create<arith::ConstantIndexOp>(loc, nTileSize);

      auto iLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, iUpper, oneIdx);
      rewriter.setInsertionPointToStart(iLoop.getBody());
      Value iIv = iLoop.getInductionVar();

      auto jLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, jUpper, oneIdx);
      rewriter.setInsertionPointToStart(jLoop.getBody());
      Value jIv = jLoop.getInductionVar();

      Value acc =
          rewriter.create<memref::LoadOp>(loc, cTile, ValueRange{iIv, jIv});
      Value part =
          rewriter.create<memref::LoadOp>(loc, partial, ValueRange{iIv, jIv});
      Value sum = rewriter.create<arith::AddFOp>(loc, acc, part);
      rewriter.create<memref::StoreOp>(loc, sum, cTile, ValueRange{iIv, jIv});

      rewriter.setInsertionPointAfter(kLoop);
      rewriter.create<memref::DeallocOp>(loc, partial);
    }

    rewriter.setInsertionPointAfter(outerLoop);

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

void mlir::buddy::populateMatrixTileMatMulPatterns(RewritePatternSet &patterns,
                                                   int64_t bankWidthBytes,
                                                   int64_t bankDepth,
                                                   int64_t bankNum) {
  patterns.add<TileMatMulLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
}
