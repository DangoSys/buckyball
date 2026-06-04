//====- LowerTileToBuckyball.cpp - Tile to Buckyball Lowering Pass -------===//
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
//
// This file defines the pass to lower Tile dialect to Buckyball dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"

#include "Utils/BankUtils.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Helper: ceil division
//===----------------------------------------------------------------------===//

static size_t ceilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

static constexpr size_t kDefaultBankWidthBytes = 16;
static constexpr size_t kMatmulTile = 16;

static size_t elemsPerBankRow(Type elemType, size_t bankWidthBytes) {
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return 0;
  return bankWidthBytes / (bitWidth / 8);
}

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(v));
}

static Value packF32BitsAsI64(OpBuilder &b, Location loc, Value f32Val) {
  Value i32Bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), f32Val);
  return b.create<arith::ExtUIOp>(loc, b.getI64Type(), i32Bits);
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

// Matches `BuckyballMatMulLowering` mvout depthC = M * (N/16) on C (i32 acc,
// cols=4). Spike/bebop: BANK_SIZE / (cols*16) = 16384/64 = 256 lines per mvout.
static constexpr size_t kMaxAccMvoutDepthLines = 256;

static size_t cMvoutDepthLines(size_t mEl, size_t nEl) {
  return mEl * (nEl / 16);
}

// `BuckyballMatMulLowering` mvin: depthA = M*(K/16), depthB = K*(N/16); i8 bank
// line_bytes=16. Spike/bebop: BANK_SIZE/16 = 1024 lines per mvin.
static constexpr size_t kMaxI8MvinDepthLines = 1024;

static size_t aMvinDepthLines(size_t mEl, size_t kEl) {
  return mEl * (kEl / 16);
}

static size_t bMvinDepthLines(size_t kEl, size_t nEl) {
  return kEl * (nEl / 16);
}

//===----------------------------------------------------------------------===//
// Tile Matmul Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileMatMulLowering : public OpRewritePattern<tile::TileMatMulOp> {
  // Compute bank rows needed: A occupies mTileLen*kTileLen, B occupies
  // kTileLen*nTileLen
  size_t computeBankRows(size_t mTileLen, size_t nTileLen,
                         size_t kTileLen) const {
    return mTileLen * kTileLen + kTileLen * nTileLen;
  }

public:
  explicit TileMatMulLowering(MLIRContext *context, int64_t /*bankWidthBytes*/,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileMatMulOp tileMatMulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileMatMulOp.getLoc();

    Value aMemArray = tileMatMulOp.getAMemArray();
    Value bMemArray = tileMatMulOp.getBMemArray();
    Value cMemArray = tileMatMulOp.getCMemArray();

    auto aType = cast<MemRefType>(aMemArray.getType());
    auto bType = cast<MemRefType>(bMemArray.getType());
    auto cType = cast<MemRefType>(cMemArray.getType());

    // A[M][K], B[K][N], C[M][N]
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

    // Buckyball matmul consumes Mx16 @ 16xN tiles. Pad at tile level so the
    // lower Buckyball layer only sees regular tiles.
    size_t M_pad = ceilDiv(M, 16) * 16;
    size_t K_pad = ceilDiv(K, 16) * 16;
    size_t N_pad = ceilDiv(N, 16) * 16;
    bool needPadding = (M_pad != M) || (K_pad != K) || (N_pad != N);

    Value aMemArrayPadded = aMemArray;
    Value bMemArrayPadded = bMemArray;
    Value cMemArrayPadded = cMemArray;

    if (needPadding) {
      auto elemType = aType.getElementType();

      // Allocate padded buffers
      auto aPadType =
          MemRefType::get({(int64_t)M_pad, (int64_t)K_pad}, elemType);
      auto bPadType =
          MemRefType::get({(int64_t)K_pad, (int64_t)N_pad}, elemType);
      auto cPadType =
          MemRefType::get({(int64_t)M_pad, (int64_t)N_pad}, elemType);

      aMemArrayPadded = rewriter.create<memref::AllocOp>(loc, aPadType);
      bMemArrayPadded = rewriter.create<memref::AllocOp>(loc, bPadType);
      cMemArrayPadded = rewriter.create<memref::AllocOp>(loc, cPadType);

      // Zero-fill padded buffers
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getZeroAttr(elemType));
      rewriter.create<linalg::FillOp>(loc, zero, aMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, zero, bMemArrayPadded);
      rewriter.create<linalg::FillOp>(loc, zero, cMemArrayPadded);

      // Copy original data to padded buffers (only valid region)
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

    // Update dimensions for tiling (use padded dimensions if needed)
    size_t M_tiling = needPadding ? M_pad : M;
    size_t K_tiling = needPadding ? K_pad : K;
    size_t N_tiling = needPadding ? N_pad : N;

    // Buckyball matmul ISA tile is 16xK @ Kx16 -> 16x16.
    const size_t mMeta = kMatmulTile;
    const size_t nMeta = kMatmulTile;
    const size_t kMeta = kMatmulTile;

    // Pad dimensions to multiples of meta lengths
    const size_t mPad = ceilDiv(M_tiling, mMeta) * mMeta;
    const size_t nPad = ceilDiv(N_tiling, nMeta) * nMeta;
    const size_t kPad = ceilDiv(K_tiling, kMeta) * kMeta;

    // Tile lengths: grow K first so kTileSize is fixed when checking mvin B
    // depth on N; then N (c mvout + mvin B), then M (mvin A). Order avoids
    // oversized depthA/B. The base 16xK and Kx16 tiles must also fit physical
    // bank mvin depth; otherwise the generated buckyball.matmul would lower to
    // illegal mvin depths before N/M growth is considered.
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

    // Generate tiled computation using scf.for loops (runtime iteration)
    // instead of C++ unrolling. The previous unrolled version generated
    // mTileNum*nTileNum*kTileNum buckyball.MatMulOps at compile time —
    // 4096 ops / 77K+ instructions for 1024x1024 inputs.
    //
    // Each buckyball.MatMulOp computes a complete K tile. When K is split,
    // accumulate those partial fp32 tiles explicitly at the Tile layer so the
    // lower Buckyball layer can keep its single-tile overwrite semantics.
    //
    // Requires mPad/nPad/kPad to be exact multiples of tile sizes.
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
      // Outer-to-inner: k -> m -> n (preserves original C++ loop nesting order)
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

      rewriter.create<buckyball::MatMulOp>(loc, aTile, bTile, cTile);
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

      rewriter.create<buckyball::MatMulOp>(loc, aTile, bTile, partial);

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

    // Copy back C from padded buffer to original output
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

      // Deallocate padded buffers
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

//===----------------------------------------------------------------------===//
// Tile Transpose Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  explicit TileTransposeLowering(MLIRContext *context, int64_t bankWidthBytes,
                                 int64_t /*bankDepth*/, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes) {}

  LogicalResult matchAndRewrite(tile::TileTransposeOp tileTransposeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileTransposeOp.getLoc();

    Value inputMemArray = tileTransposeOp.getAMemArray();
    Value outputMemArray = tileTransposeOp.getBMemArray();

    auto inputType = cast<MemRefType>(inputMemArray.getType());
    auto outputType = cast<MemRefType>(outputMemArray.getType());
    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();

    size_t Rows = inShape[inShape.size() - 2];
    size_t Cols = inShape[inShape.size() - 1];

    if (outShape[outShape.size() - 2] != (int64_t)Cols ||
        outShape[outShape.size() - 1] != (int64_t)Rows)
      return tileTransposeOp.emitError(
          "Output shape must be transposed of input shape");

    size_t elemsPerRow =
        elemsPerBankRow(inputType.getElementType(), bankWidthBytes);
    if (elemsPerRow == 0)
      return tileTransposeOp.emitError("unsupported transpose element type");

    // Hardware constraint: transpose processes 16 rows at a time
    // iter parameter: number of columns (max 64 for i8)
    constexpr size_t kTransposeRows = kMatmulTile;
    constexpr size_t kMaxTransposeCols = 64;

    // Tile columns to fit hardware limit
    size_t colTileSize = std::min(Cols, kMaxTransposeCols);
    // Align to physical bank rows for efficient mvin/mvout
    colTileSize = (colTileSize / elemsPerRow) * elemsPerRow;
    if (colTileSize == 0)
      colTileSize = elemsPerRow;

    size_t rowTileNum = ceilDiv(Rows, kTransposeRows);
    size_t colTileNum = ceilDiv(Cols, colTileSize);

    for (size_t r0 = 0; r0 < rowTileNum; r0++) {
      for (size_t c0 = 0; c0 < colTileNum; c0++) {
        size_t rStart = r0 * kTransposeRows;
        size_t cStart = c0 * colTileSize;
        size_t rLen = std::min(kTransposeRows, Rows - rStart);
        size_t cLen = std::min(colTileSize, Cols - cStart);

        // Hardware requires exactly 16 rows; pad if needed
        size_t rLenPadded = (rLen < kTransposeRows) ? kTransposeRows : rLen;

        Value inTile = rewriter.create<memref::SubViewOp>(
            loc, inputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rStart),
                                      rewriter.getIndexAttr(cStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rLen),
                                      rewriter.getIndexAttr(cLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, outputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cStart),
                                      rewriter.getIndexAttr(rStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cLen),
                                      rewriter.getIndexAttr(rLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        // Allocate banks: use default (row=1, col=1) for i8 data
        Value srcBank =
            rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
        Value dstBank =
            rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());

        // Move data from memref to source bank
        // depth = number of physical bank rows
        int64_t depth = rLenPadded * cLen / elemsPerRow;
        Value srcBankAfterMvin =
            buckyball::mvinBank(rewriter, loc, inTile, srcBank, depth);

        // Execute transpose: iter = number of columns
        Value iterVal = buckyball::createI64Const(rewriter, loc, cLen);
        Value modeVal = buckyball::createI64Const(rewriter, loc, 0);
        Value dstBankAfterTranspose =
            rewriter.create<buckyball::BankTransposeOp>(
                loc, dstBank.getType(), srcBankAfterMvin, dstBank, iterVal,
                modeVal);

        // Move result from destination bank to memref
        // Output is cLen × rLenPadded, but we only mvout cLen × rLen
        int64_t outDepth = cLen * rLen / elemsPerRow;
        buckyball::mvoutBank(rewriter, loc, outTile, dstBankAfterTranspose,
                             outDepth);

        // Release banks
        buckyball::releaseBank(rewriter, loc, srcBankAfterMvin);
        buckyball::releaseBank(rewriter, loc, dstBankAfterTranspose);
      }
    }

    rewriter.eraseOp(tileTransposeOp);
    return success();
  }

private:
  int64_t bankWidthBytes;
};

} // namespace

//===----------------------------------------------------------------------===//
// Tile Conv2d Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  explicit TileConv2dLowering(MLIRContext *context, int64_t bankWidthBytes,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes),
        bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileConv2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getInput();   // [N, H, W, C]
    Value filter = op.getFilter(); // [KH, KW, C, OC]
    Value output = op.getOutput(); // [N, OH, OW, OC]

    auto inType = cast<MemRefType>(input.getType());
    auto filterType = cast<MemRefType>(filter.getType());
    auto outType = cast<MemRefType>(output.getType());

    auto inShape = inType.getShape();
    auto fShape = filterType.getShape();
    auto outShape = outType.getShape();

    int64_t N = inShape[0], H = inShape[1], W = inShape[2], C = inShape[3];
    int64_t KH = fShape[0], KW = fShape[1], OC = fShape[3];
    int64_t OH = outShape[1], OW = outShape[2];

    int64_t totalOHOW = OH * OW;
    int64_t patchCols = KH * KW * C;
    int64_t i8ElemsPerRow = bankWidthBytes;
    if (!inType.getElementType().isF32() ||
        !filterType.getElementType().isF32())
      return op.emitError("tile_conv2d im2col lowering currently expects f32");
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0 || KH <= 0 || KW <= 0 || OC <= 0 ||
        OH <= 0 || OW <= 0)
      return op.emitError("tile_conv2d requires positive static shapes");
    if (patchCols <= 0 || patchCols > (int64_t)bankDepth)
      return op.emitError("tile_conv2d patch size exceeds bank depth");
    if (KH > 255 || KW * C > 255 || H > 255 || W * C > 255 || OH > 255 ||
        OW > 255 || C > 255)
      return op.emitError("tile_conv2d im2col shape exceeds 8-bit ISA fields");
    if ((H * W * C) % i8ElemsPerRow != 0)
      return op.emitError(
          "tile_conv2d input element count must align to bank width");
    (void)totalOHOW;

    // For each batch
    for (int64_t n = 0; n < N; n++) {
      Value inBatch = rewriter.create<memref::SubViewOp>(
          loc, input,
          SmallVector<OpFoldResult>{
              rewriter.getIndexAttr(n), rewriter.getIndexAttr(0),
              rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{
              rewriter.getIndexAttr(1), rewriter.getIndexAttr(H),
              rewriter.getIndexAttr(W), rewriter.getIndexAttr(C)},
          SmallVector<OpFoldResult>{
              rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
              rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
      auto collapseIn = rewriter.create<memref::CollapseShapeOp>(
          loc, inBatch, SmallVector<ReassociationIndices>{{0, 1}, {2, 3}});

      Value filterReshaped = rewriter.create<memref::CollapseShapeOp>(
          loc, filter, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
      int64_t ocPadded = ceilDiv(OC, kMatmulTile) * kMatmulTile;
      auto filterPadType =
          MemRefType::get({patchCols, ocPadded}, filterType.getElementType());
      auto filterPadAlloc =
          rewriter.create<memref::AllocOp>(loc, filterPadType);
      filterPadAlloc->setAttr("alignment", rewriter.getI64IntegerAttr(16));
      Value filterPad = filterPadAlloc.getResult();
      Value zeroF32 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0));
      rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                      ValueRange{filterPad});

      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value kUpper = rewriter.create<arith::ConstantIndexOp>(loc, patchCols);
      Value ocUpper = rewriter.create<arith::ConstantIndexOp>(loc, OC);
      auto kLoop = rewriter.create<scf::ForOp>(loc, zero, kUpper, one);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        auto ocLoop = rewriter.create<scf::ForOp>(loc, zero, ocUpper, one);
        rewriter.setInsertionPointToStart(ocLoop.getBody());
        Value v = rewriter.create<memref::LoadOp>(
            loc, filterReshaped,
            ValueRange{kLoop.getInductionVar(), ocLoop.getInductionVar()});
        rewriter.create<memref::StoreOp>(
            loc, v, filterPad,
            ValueRange{kLoop.getInductionVar(), ocLoop.getInductionVar()});
      }

      Value outBatch = rewriter.create<memref::SubViewOp>(
          loc, output,
          SmallVector<OpFoldResult>{
              rewriter.getIndexAttr(n), rewriter.getIndexAttr(0),
              rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{
              rewriter.getIndexAttr(1), rewriter.getIndexAttr(OH),
              rewriter.getIndexAttr(OW), rewriter.getIndexAttr(OC)},
          SmallVector<OpFoldResult>{
              rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
              rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
      auto collapseOut = rewriter.create<memref::CollapseShapeOp>(
          loc, outBatch, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});

      Value inputFp = buckyball::allocBank(rewriter, loc, 1, 4);
      Value inputI8 = buckyball::allocBank(rewriter, loc, 1, 1);

      int64_t inputDepth = H * W * C / i8ElemsPerRow;
      Value inputLoad =
          buckyball::mvinBank(rewriter, loc, collapseIn, inputFp, inputDepth);
      Value inputMax = buildTileAbsMax(rewriter, loc, collapseIn, H, W * C);
      Value inputScale = buildQuantScale(rewriter, loc, inputMax);
      Value inputScaleBits = packF32BitsAsI64(rewriter, loc, inputScale);
      Value inputQuant = rewriter.create<buckyball::BankFp2IntOp>(
          loc, inputI8.getType(), inputLoad, inputI8,
          buckyball::createI64Const(rewriter, loc, inputDepth), inputScaleBits);
      buckyball::releaseBank(rewriter, loc, inputLoad);

      for (int64_t oc0 = 0; oc0 < OC; oc0 += kMatmulTile) {
        Value oc0Idx = rewriter.create<arith::ConstantIndexOp>(loc, oc0);
        Value filterTile = rewriter.create<memref::SubViewOp>(
            loc, filterPad,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), oc0Idx},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(patchCols),
                                      rewriter.getIndexAttr(kMatmulTile)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value filterFp = buckyball::allocBank(rewriter, loc, 1, 4);
        Value filterI8 = buckyball::allocBank(rewriter, loc, 1, 1);
        Value filterMax =
            buildTileAbsMax(rewriter, loc, filterTile, patchCols, kMatmulTile);
        Value filterScale = buildQuantScale(rewriter, loc, filterMax);
        Value filterScaleBits = packF32BitsAsI64(rewriter, loc, filterScale);
        Value filterLoad =
            buckyball::mvinBank(rewriter, loc, filterTile, filterFp, patchCols);
        Value filterQuant = rewriter.create<buckyball::BankFp2IntOp>(
            loc, filterI8.getType(), filterLoad, filterI8,
            buckyball::createI64Const(rewriter, loc, patchCols),
            filterScaleBits);
        buckyball::releaseBank(rewriter, loc, filterLoad);

        for (int64_t oh0 = 0; oh0 < OH; ++oh0) {
          for (int64_t ow0 = 0; ow0 < OW; ow0 += kMatmulTile) {
            int64_t mLen = std::min<int64_t>(kMatmulTile, OW - ow0);
            int64_t outOffset = oh0 * OW + ow0;

            auto cTileType =
                MemRefType::get({(int64_t)kMatmulTile, (int64_t)kMatmulTile},
                                outType.getElementType());
            auto cTileAlloc = rewriter.create<memref::AllocOp>(loc, cTileType);
            cTileAlloc->setAttr("alignment", rewriter.getI64IntegerAttr(16));
            Value cTile = cTileAlloc.getResult();

            Value patchI8 = buckyball::allocBank(rewriter, loc, 1, 1);
            Value patchT = buckyball::allocBank(rewriter, loc, 1, 1);
            Value cI32 = buckyball::allocBank(rewriter, loc, 1, 4);

            Value patch = rewriter.create<buckyball::BankIm2colOp>(
                loc, patchI8.getType(), inputQuant, patchI8,
                buckyball::createI64Const(rewriter, loc, KH),
                buckyball::createI64Const(rewriter, loc, KW * C),
                buckyball::createI64Const(rewriter, loc, H),
                buckyball::createI64Const(rewriter, loc, W * C),
                buckyball::createI64Const(rewriter, loc, oh0),
                buckyball::createI64Const(rewriter, loc, ow0 * C),
                buckyball::createI64Const(rewriter, loc, C));

            Value patchTrans = rewriter.create<buckyball::BankTransposeOp>(
                loc, patchT.getType(), patch, patchT,
                buckyball::createI64Const(rewriter, loc, patchCols),
                buckyball::createI64Const(rewriter, loc, 0));
            buckyball::releaseBank(rewriter, loc, patch);

            Value cMul = rewriter.create<buckyball::BankMulWarp16Op>(
                loc, cI32.getType(), patchTrans, filterQuant, cI32,
                buckyball::createI64Const(rewriter, loc, patchCols),
                buckyball::createI64Const(rewriter, loc, 0));
            buckyball::releaseBank(rewriter, loc, patchTrans);

            Value cFp32 = buckyball::allocBank(rewriter, loc, 1, 4);
            Value oneF32 = cstF32(rewriter, loc, 1.0f);
            Value scaleProd =
                rewriter.create<arith::MulFOp>(loc, inputScale, filterScale);
            Value dequantScale =
                rewriter.create<arith::DivFOp>(loc, oneF32, scaleProd);
            Value dequantScaleBits =
                packF32BitsAsI64(rewriter, loc, dequantScale);
            Value cDequant = rewriter.create<buckyball::BankInt2FpOp>(
                loc, cFp32.getType(), cMul, cFp32,
                buckyball::createI64Const(rewriter, loc, kMatmulTile),
                dequantScaleBits);
            buckyball::releaseBank(rewriter, loc, cMul);
            Value cStore = buckyball::mvoutBank(rewriter, loc, cTile, cDequant,
                                                kMatmulTile);
            rewriter.create<buckyball::FenceOp>(loc);
            buckyball::releaseBank(rewriter, loc, cStore);

            Value mUpper = rewriter.create<arith::ConstantIndexOp>(loc, mLen);
            Value cUpper = rewriter.create<arith::ConstantIndexOp>(
                loc, std::min<int64_t>(kMatmulTile, OC - oc0));
            auto mLoop = rewriter.create<scf::ForOp>(loc, zero, mUpper, one);
            {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(mLoop.getBody());
              auto cLoop = rewriter.create<scf::ForOp>(loc, zero, cUpper, one);
              rewriter.setInsertionPointToStart(cLoop.getBody());
              Value v = rewriter.create<memref::LoadOp>(
                  loc, cTile,
                  ValueRange{mLoop.getInductionVar(), cLoop.getInductionVar()});
              Value outM = rewriter.create<arith::AddIOp>(
                  loc, mLoop.getInductionVar(),
                  rewriter.create<arith::ConstantIndexOp>(loc, outOffset));
              Value outC = rewriter.create<arith::AddIOp>(
                  loc, cLoop.getInductionVar(), oc0Idx);
              rewriter.create<memref::StoreOp>(loc, v, collapseOut,
                                               ValueRange{outM, outC});
            }

            rewriter.create<memref::DeallocOp>(loc, cTile);
          }
        }

        buckyball::releaseBank(rewriter, loc, filterQuant);
      }

      buckyball::releaseBank(rewriter, loc, inputQuant);
      rewriter.create<memref::DeallocOp>(loc, filterPad);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t bankWidthBytes, bankDepth;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void populateLowerTileToBuckyballConversionPatterns(RewritePatternSet &patterns,
                                                    int64_t bankWidthBytes,
                                                    int64_t bankDepth,
                                                    int64_t bankNum) {
  patterns.add<TileMatMulLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
  patterns.add<TileTransposeLowering>(patterns.getContext(), bankWidthBytes,
                                      bankDepth, bankNum);
  patterns.add<TileConv2dLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
}

//===----------------------------------------------------------------------===//
// LowerTileToBuckyball Pass
//===----------------------------------------------------------------------===//

namespace {
class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)
  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert Tile dialect to Buckyball dialect";
  }
  LowerTileToBuckyballPass() = default;
  LowerTileToBuckyballPass(const LowerTileToBuckyballPass &) {}

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(kDefaultBankWidthBytes)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Bank depth (rows per bank)."),
                            llvm::cl::init(4096)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tile::TileDialect, buckyball::BuckyballDialect,
                func::FuncDialect, memref::MemRefDialect, arith::ArithDialect,
                scf::SCFDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerTileToBuckyballPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<buckyball::BuckyballDialect, memref::MemRefDialect,
                         arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect, linalg::LinalgDialect>();
  target.addIllegalDialect<tile::TileDialect>();

  RewritePatternSet patterns(context);
  populateLowerTileToBuckyballConversionPatterns(patterns, bankWidthBytes,
                                                 bankDepth, bankNum);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
} // namespace buddy
} // namespace mlir
