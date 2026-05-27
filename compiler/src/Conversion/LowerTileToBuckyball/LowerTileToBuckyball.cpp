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
  explicit TileMatMulLowering(MLIRContext *context, int64_t lane, int64_t warp,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), lane(lane), warp(warp),
        bankDepth(bankDepth) {}

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

    // Block counts per axis: each unit is lane×lane×warp elements (see
    // mTileSize below).
    const size_t mMeta = lane;
    const size_t nMeta = lane;
    const size_t kMeta = warp;

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

    while ((kTileLen + 1) * kMeta <= kPad &&
           computeBankRows(1, 1, kTileLen + 1) <= (size_t)bankDepth &&
           aMvinDepthLines(mMeta, (kTileLen + 1) * kMeta) <=
               kMaxI8MvinDepthLines &&
           bMvinDepthLines((kTileLen + 1) * kMeta, nMeta) <=
               kMaxI8MvinDepthLines)
      kTileLen++;

    const size_t kTileSize = kTileLen * kMeta;

    while ((nTileLen + 1) * nMeta <= nPad &&
           computeBankRows(1, nTileLen + 1, kTileLen) <= (size_t)bankDepth &&
           cMvoutDepthLines(mMeta, (nTileLen + 1) * nMeta) <=
               kMaxAccMvoutDepthLines &&
           bMvinDepthLines(kTileSize, (nTileLen + 1) * nMeta) <=
               kMaxI8MvinDepthLines)
      nTileLen++;

    while ((mTileLen + 1) * mMeta <= mPad &&
           computeBankRows(mTileLen + 1, nTileLen, kTileLen) <=
               (size_t)bankDepth &&
           cMvoutDepthLines((mTileLen + 1) * mMeta, nTileLen * nMeta) <=
               kMaxAccMvoutDepthLines &&
           aMvinDepthLines((mTileLen + 1) * mMeta, kTileSize) <=
               kMaxI8MvinDepthLines)
      mTileLen++;

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
  int64_t lane, warp, bankDepth;
};

} // namespace

//===----------------------------------------------------------------------===//
// Tile Transpose Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  explicit TileTransposeLowering(MLIRContext *context, int64_t lane,
                                 int64_t /*warp*/, int64_t /*bankDepth*/,
                                 int64_t /*bankNum*/)
      : OpRewritePattern(context), lane(lane) {}

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

    // Hardware constraint: transpose processes 16 rows at a time
    // iter parameter: number of columns (max 64 for i8)
    constexpr size_t kTransposeRows = 16;
    constexpr size_t kMaxTransposeCols = 64;

    // Tile columns to fit hardware limit
    size_t colTileSize = std::min(Cols, kMaxTransposeCols);
    // Align to lane for efficient mvin/mvout
    colTileSize = (colTileSize / lane) * lane;
    if (colTileSize == 0)
      colTileSize = lane;

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
        // depth = number of lane-width rows = rLenPadded * cLen / lane
        int64_t depth = rLenPadded * cLen / lane;
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
        int64_t outDepth = cLen * rLen / lane;
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
  int64_t lane;
};

} // namespace

//===----------------------------------------------------------------------===//
// Tile Conv2d Lowering Pattern
//===----------------------------------------------------------------------===//

namespace {

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  explicit TileConv2dLowering(MLIRContext *context, int64_t lane,
                              int64_t /*warp*/, int64_t bankDepth,
                              int64_t /*bankNum*/)
      : OpRewritePattern(context), lane(lane), bankDepth(bankDepth) {}

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

    IntegerType i64Type = rewriter.getI64Type();

    // Im2col patch dimensions: patchRows = OH*OW, patchCols = KH*KW*C
    int64_t patchCols = KH * KW * C;

    // Pad patchCols to lane boundary for matmul
    int64_t patchColsPad = ceilDiv(patchCols, (int64_t)lane) * lane;

    // Tile OH*OW dimension: how many output rows per tile
    // Each tile produces tileOHOW output pixels, requiring tileOHOW rows in
    // patch matrix Bank constraint: patch tile (tileOHOW * patchColsPad) must
    // fit in bank
    int64_t tileOHOW =
        std::min((int64_t)bankDepth / std::max(patchColsPad, (int64_t)1),
                 (int64_t)(OH * OW));
    if (tileOHOW < 1)
      tileOHOW = 1;
    // Align to lane boundary
    tileOHOW = (tileOHOW / lane) * lane;
    if (tileOHOW < lane)
      tileOHOW = lane;

    int64_t totalOHOW = OH * OW;
    int64_t tileNum = ceilDiv(totalOHOW, tileOHOW);

    // For each batch
    for (int64_t n = 0; n < N; n++) {
      for (int64_t t = 0; t < tileNum; t++) {
        int64_t ohowStart = t * tileOHOW;
        int64_t ohowLen = std::min(tileOHOW, totalOHOW - ohowStart);

        // Compute start row/col in input space for im2col
        int64_t startRow = (ohowStart / OW); // starting OH index
        int64_t startCol = (ohowStart % OW); // starting OW index

        // Allocate temporary patch matrix: [ohowLen, patchColsPad]
        auto elemType = inType.getElementType();
        auto patchType = MemRefType::get({ohowLen, patchColsPad}, elemType);
        Value patchBuf = rewriter.create<memref::AllocOp>(loc, patchType);

        // Create subview of input for batch n, then collapse to 2D for im2col
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

        // Collapse [1, H, W, C] → [H, W*C] for im2col input
        auto collapseIn = rewriter.create<memref::CollapseShapeOp>(
            loc, inBatch, SmallVector<ReassociationIndices>{{0, 1}, {2, 3}});

        // Im2col: rearrange input patches into columns
        Value kRowVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(KH));
        Value kColVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(KW));
        Value inRowVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(H));
        Value inColVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(W * C));
        Value startRowVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(startRow));
        Value startColVal = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(startCol));

        // Allocate source and destination banks
        Value srcBank = buckyball::allocBank(rewriter, loc, H, W * C / lane);
        Value dstBank =
            buckyball::allocBank(rewriter, loc, ohowLen, patchColsPad / lane);

        // Move input data to source bank
        int64_t inDepth = H * W * C / lane;
        Value srcBankAfterMvin =
            buckyball::mvinBank(rewriter, loc, collapseIn, srcBank, inDepth);

        // Execute im2col operation
        Value dstBankAfterIm2col = rewriter.create<buckyball::BankIm2colOp>(
            loc, dstBank.getType(), srcBankAfterMvin, dstBank, kRowVal, kColVal,
            inRowVal, inColVal, startRowVal, startColVal);

        // Move result to patch buffer
        int64_t outDepth = ohowLen * patchColsPad / lane;
        buckyball::mvoutBank(rewriter, loc, patchBuf, dstBankAfterIm2col,
                             outDepth);

        // Release banks
        buckyball::releaseBank(rewriter, loc, srcBankAfterMvin);
        buckyball::releaseBank(rewriter, loc, dstBankAfterIm2col);

        // Reshape filter [KH, KW, C, OC] → [KH*KW*C, OC] for matmul
        Value filterReshaped = rewriter.create<memref::CollapseShapeOp>(
            loc, filter, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});

        // Create output subview for this tile: [ohowLen, OC]
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

        // Collapse [1, OH, OW, OC] → [OH*OW, OC]
        auto collapseOut = rewriter.create<memref::CollapseShapeOp>(
            loc, outBatch, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});

        // Subview for the current tile rows
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, collapseOut,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(ohowStart),
                                      rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(ohowLen),
                                      rewriter.getIndexAttr(OC)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        // MatMul: patch[ohowLen, patchCols] x filter[patchCols, OC] →
        // out[ohowLen, OC]
        rewriter.create<buckyball::MatMulOp>(loc, patchBuf, filterReshaped,
                                             outTile);

        // Free temporary buffer
        rewriter.create<memref::DeallocOp>(loc, patchBuf);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t lane, bankDepth;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void populateLowerTileToBuckyballConversionPatterns(RewritePatternSet &patterns,
                                                    int64_t lane, int64_t warp,
                                                    int64_t bankDepth,
                                                    int64_t bankNum) {
  patterns.add<TileMatMulLowering>(patterns.getContext(), lane, warp, bankDepth,
                                   bankNum);
  patterns.add<TileTransposeLowering>(patterns.getContext(), lane, warp,
                                      bankDepth, bankNum);
  patterns.add<TileConv2dLowering>(patterns.getContext(), lane, warp, bankDepth,
                                   bankNum);
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

  Option<int64_t> lane{*this, "lane", llvm::cl::desc("Hardware lane width."),
                       llvm::cl::init(16)};
  Option<int64_t> warp{*this, "warp", llvm::cl::desc("Warp depth."),
                       llvm::cl::init(16)};
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
  populateLowerTileToBuckyballConversionPatterns(patterns, lane, warp,
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
