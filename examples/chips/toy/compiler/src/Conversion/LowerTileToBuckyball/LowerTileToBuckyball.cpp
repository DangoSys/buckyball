//===- LowerTileToBuckyball.cpp - Toy tile->buckyball pass ---------------===//
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
#include "Tile/Transform.h"

#include "Utils/BankUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::tile;
namespace tile = ::buddy::tile;
using mlir::buddy::ceilDiv;
using mlir::buddy::kDefaultBankWidthBytes;
using mlir::buddy::kMatmulTile;
using mlir::buddy::populateMatrixTileMatMulPatterns;

namespace {

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

    constexpr size_t kTransposeRows = kMatmulTile;
    constexpr size_t kMaxTransposeCols = 64;

    size_t colTileSize = std::min(Cols, kMaxTransposeCols);
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

        Value srcBank =
            rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());
        Value dstBank =
            rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());

        int64_t depth = rLenPadded * cLen / elemsPerRow;
        Value srcBankAfterMvin =
            mvinBank(rewriter, loc, inTile, srcBank, depth);

        Value iterVal = createI64Const(rewriter, loc, cLen);
        Value modeVal = createI64Const(rewriter, loc, 0);
        Value dstBankAfterTranspose = rewriter.create<BankTransposeOp>(
            loc, dstBank.getType(), srcBankAfterMvin, dstBank, iterVal,
            modeVal);

        int64_t outDepth = cLen * rLen / elemsPerRow;
        mvoutBank(rewriter, loc, outTile, dstBankAfterTranspose, outDepth);

        releaseBank(rewriter, loc, srcBankAfterMvin);
        releaseBank(rewriter, loc, dstBankAfterTranspose);
      }
    }

    rewriter.eraseOp(tileTransposeOp);
    return success();
  }

private:
  int64_t bankWidthBytes;
};

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  explicit TileConv2dLowering(MLIRContext *context, int64_t bankWidthBytes,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes),
        bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileConv2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getInput();
    Value filter = op.getFilter();
    Value output = op.getOutput();

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
    int64_t i8ElemsPerRow = bankWidthBytes;
    bool needInputPad = (H * W * C) % i8ElemsPerRow != 0;
    int64_t cPadded = C;
    while ((H * W * cPadded) % i8ElemsPerRow != 0)
      ++cPadded;
    int64_t patchCols = KH * KW * cPadded;
    if (!inType.getElementType().isF32() ||
        !filterType.getElementType().isF32())
      return op.emitError("tile_conv2d im2col lowering currently expects f32");
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0 || KH <= 0 || KW <= 0 || OC <= 0 ||
        OH <= 0 || OW <= 0)
      return op.emitError("tile_conv2d requires positive static shapes");
    if (patchCols <= 0)
      return op.emitError("tile_conv2d requires positive patch size");
    if (patchCols > (int64_t)bankDepth)
      return op.emitError("tile_conv2d patch size exceeds bank depth");
    if (KH > 255 || KW * cPadded > 255 || H > 255 || W * cPadded > 255 ||
        OH > 255 || OW > 255 || cPadded > 255)
      return op.emitError("tile_conv2d im2col shape exceeds 8-bit ISA fields");
    (void)totalOHOW;

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
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value zeroF32 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0));

      Value input2d;
      Value inputPad;
      if (needInputPad) {
        auto inputPadType =
            MemRefType::get({H, W, cPadded}, inType.getElementType());
        auto inputPadAlloc =
            rewriter.create<memref::AllocOp>(loc, inputPadType);
        inputPadAlloc->setAttr("alignment", rewriter.getI64IntegerAttr(16));
        inputPad = inputPadAlloc.getResult();
        rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                        ValueRange{inputPad});

        Value hUpper = rewriter.create<arith::ConstantIndexOp>(loc, H);
        Value wUpper = rewriter.create<arith::ConstantIndexOp>(loc, W);
        Value cUpper = rewriter.create<arith::ConstantIndexOp>(loc, C);
        auto hLoop = rewriter.create<scf::ForOp>(loc, zero, hUpper, one);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(hLoop.getBody());
          auto wLoop = rewriter.create<scf::ForOp>(loc, zero, wUpper, one);
          rewriter.setInsertionPointToStart(wLoop.getBody());
          auto cLoop = rewriter.create<scf::ForOp>(loc, zero, cUpper, one);
          rewriter.setInsertionPointToStart(cLoop.getBody());
          Value v = rewriter.create<memref::LoadOp>(
              loc, inBatch,
              ValueRange{zero, hLoop.getInductionVar(), wLoop.getInductionVar(),
                         cLoop.getInductionVar()});
          rewriter.create<memref::StoreOp>(loc, v, inputPad,
                                           ValueRange{hLoop.getInductionVar(),
                                                      wLoop.getInductionVar(),
                                                      cLoop.getInductionVar()});
        }

        input2d = rewriter.create<memref::CollapseShapeOp>(
            loc, inputPad, SmallVector<ReassociationIndices>{{0}, {1, 2}});
      } else {
        input2d = rewriter.create<memref::CollapseShapeOp>(
            loc, inBatch, SmallVector<ReassociationIndices>{{0, 1}, {2, 3}});
      }

      int64_t ocPadded = ceilDiv(OC, kMatmulTile) * kMatmulTile;
      auto filterPadType =
          MemRefType::get({patchCols, ocPadded}, filterType.getElementType());
      auto filterPadAlloc =
          rewriter.create<memref::AllocOp>(loc, filterPadType);
      filterPadAlloc->setAttr("alignment", rewriter.getI64IntegerAttr(16));
      Value filterPad = filterPadAlloc.getResult();
      rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                      ValueRange{filterPad});

      Value khUpper = rewriter.create<arith::ConstantIndexOp>(loc, KH);
      Value kwUpper = rewriter.create<arith::ConstantIndexOp>(loc, KW);
      Value cUpper = rewriter.create<arith::ConstantIndexOp>(loc, C);
      Value ocUpper = rewriter.create<arith::ConstantIndexOp>(loc, OC);
      Value kwConst = rewriter.create<arith::ConstantIndexOp>(loc, KW);
      Value cPaddedConst =
          rewriter.create<arith::ConstantIndexOp>(loc, cPadded);
      auto khLoop = rewriter.create<scf::ForOp>(loc, zero, khUpper, one);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(khLoop.getBody());
        auto kwLoop = rewriter.create<scf::ForOp>(loc, zero, kwUpper, one);
        rewriter.setInsertionPointToStart(kwLoop.getBody());
        auto cLoop = rewriter.create<scf::ForOp>(loc, zero, cUpper, one);
        rewriter.setInsertionPointToStart(cLoop.getBody());
        auto ocLoop = rewriter.create<scf::ForOp>(loc, zero, ocUpper, one);
        rewriter.setInsertionPointToStart(ocLoop.getBody());
        Value khKw = rewriter.create<arith::MulIOp>(
            loc, khLoop.getInductionVar(), kwConst);
        Value khKwPlus =
            rewriter.create<arith::AddIOp>(loc, khKw, kwLoop.getInductionVar());
        Value rowBase =
            rewriter.create<arith::MulIOp>(loc, khKwPlus, cPaddedConst);
        Value row = rewriter.create<arith::AddIOp>(loc, rowBase,
                                                   cLoop.getInductionVar());
        Value v = rewriter.create<memref::LoadOp>(
            loc, filter,
            ValueRange{khLoop.getInductionVar(), kwLoop.getInductionVar(),
                       cLoop.getInductionVar(), ocLoop.getInductionVar()});
        rewriter.create<memref::StoreOp>(
            loc, v, filterPad, ValueRange{row, ocLoop.getInductionVar()});
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

      Value inputFp = allocBank(rewriter, loc, 1, 4);
      Value inputI8 = allocBank(rewriter, loc, 1, 1);

      int64_t inputDepth = H * W * cPadded / i8ElemsPerRow;
      Value inputLoad = mvinBank(rewriter, loc, input2d, inputFp, inputDepth);
      Value inputMax = buildTileAbsMax(rewriter, loc, input2d, H, W * cPadded);
      Value inputScale = buildQuantScale(rewriter, loc, inputMax);
      Value inputScaleBits = packF32BitsAsI64(rewriter, loc, inputScale);
      Value inputQuant = rewriter.create<BankFp2IntOp>(
          loc, inputI8.getType(), inputLoad, inputI8,
          createI64Const(rewriter, loc, inputDepth), inputScaleBits);
      releaseBank(rewriter, loc, inputLoad);

      for (int64_t oc0 = 0; oc0 < OC; oc0 += kMatmulTile) {
        Value oc0Idx = rewriter.create<arith::ConstantIndexOp>(loc, oc0);
        Value filterTile = rewriter.create<memref::SubViewOp>(
            loc, filterPad,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), oc0Idx},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(patchCols),
                                      rewriter.getIndexAttr(kMatmulTile)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value filterFp = allocBank(rewriter, loc, 1, 4);
        Value filterI8 = allocBank(rewriter, loc, 1, 1);
        Value filterMax =
            buildTileAbsMax(rewriter, loc, filterTile, patchCols, kMatmulTile);
        Value filterScale = buildQuantScale(rewriter, loc, filterMax);
        Value filterScaleBits = packF32BitsAsI64(rewriter, loc, filterScale);
        Value filterLoad =
            mvinBank(rewriter, loc, filterTile, filterFp, patchCols);
        Value filterQuant = rewriter.create<BankFp2IntOp>(
            loc, filterI8.getType(), filterLoad, filterI8,
            createI64Const(rewriter, loc, patchCols), filterScaleBits);
        releaseBank(rewriter, loc, filterLoad);

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

            Value patchI8 = allocBank(rewriter, loc, 1, 1);
            Value patchT = allocBank(rewriter, loc, 1, 1);
            Value cI32 = allocBank(rewriter, loc, 1, 4);

            Value patch = rewriter.create<BankIm2colOp>(
                loc, patchI8.getType(), inputQuant, patchI8,
                createI64Const(rewriter, loc, KH),
                createI64Const(rewriter, loc, KW * cPadded),
                createI64Const(rewriter, loc, H),
                createI64Const(rewriter, loc, W * cPadded),
                createI64Const(rewriter, loc, oh0),
                createI64Const(rewriter, loc, ow0 * cPadded),
                createI64Const(rewriter, loc, cPadded));

            Value patchTrans = rewriter.create<BankTransposeOp>(
                loc, patchT.getType(), patch, patchT,
                createI64Const(rewriter, loc, patchCols),
                createI64Const(rewriter, loc, 0));
            releaseBank(rewriter, loc, patch);

            Value cMul = rewriter.create<BankMulWarp16Op>(
                loc, cI32.getType(), patchTrans, filterQuant, cI32,
                createI64Const(rewriter, loc, patchCols),
                createI64Const(rewriter, loc, 0));
            releaseBank(rewriter, loc, patchTrans);

            Value cFp32 = allocBank(rewriter, loc, 1, 4);
            Value oneF32 = cstF32(rewriter, loc, 1.0f);
            Value scaleProd =
                rewriter.create<arith::MulFOp>(loc, inputScale, filterScale);
            Value dequantScale =
                rewriter.create<arith::DivFOp>(loc, oneF32, scaleProd);
            Value dequantScaleBits =
                packF32BitsAsI64(rewriter, loc, dequantScale);
            Value cDequant = rewriter.create<BankInt2FpOp>(
                loc, cFp32.getType(), cMul, cFp32,
                createI64Const(rewriter, loc, kMatmulTile), dequantScaleBits);
            releaseBank(rewriter, loc, cMul);
            Value cStore =
                mvoutBank(rewriter, loc, cTile, cDequant, kMatmulTile);
            rewriter.create<FenceOp>(loc);
            releaseBank(rewriter, loc, cStore);

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

        releaseBank(rewriter, loc, filterQuant);
      }

      releaseBank(rewriter, loc, inputQuant);
      if (needInputPad)
        rewriter.create<memref::DeallocOp>(loc, inputPad);
      rewriter.create<memref::DeallocOp>(loc, filterPad);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t bankWidthBytes, bankDepth;
};

void populateToyLocalTilePatterns(RewritePatternSet &patterns,
                                  int64_t bankWidthBytes, int64_t bankDepth,
                                  int64_t bankNum) {
  patterns.add<TileTransposeLowering>(patterns.getContext(), bankWidthBytes,
                                      bankDepth, bankNum);
  patterns.add<TileConv2dLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
}

} // namespace

void mlir::populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  populateMatrixTileMatMulPatterns(patterns, bankWidthBytes, bankDepth,
                                   bankNum);
  populateToyLocalTilePatterns(patterns, bankWidthBytes, bankDepth, bankNum);
}

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
    registry.insert<::buddy::tile::TileDialect,
                    ::buddy::buckyball::BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<::buddy::buckyball::BuckyballDialect,
                           memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
    target.addIllegalDialect<::buddy::tile::TileDialect>();

    RewritePatternSet patterns(context);
    populateLowerTileToBuckyballConversionPatterns(patterns, bankWidthBytes,
                                                   bankDepth, bankNum);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::buddy::registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
