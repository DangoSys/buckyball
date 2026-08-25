//===- LowerTileToBuckyball.cpp - Pebble tile->buckyball pass -------------===//
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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"
#include "Tile/Transform.h"

#include <climits>

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;
using mlir::buddy::kDefaultBankWidthBytes;
using mlir::buddy::populateSMatMulBallTileLoweringPatterns;

namespace {

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileTransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto outputType = dyn_cast<MemRefType>(op.getBMemArray().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return op.emitError("requires static input and output memrefs");
    if (inputType.getRank() != 2 || outputType.getRank() != 2)
      return op.emitError("requires rank-2 memrefs");
    if (outputType.getShape()[0] != inputType.getShape()[1] ||
        outputType.getShape()[1] != inputType.getShape()[0])
      return op.emitError("output shape must transpose the input shape");
    if (inputType.getElementType() != outputType.getElementType())
      return op.emitError("input/output element types must match");

    rewriter.create<MemTransposeOp>(op.getLoc(), op.getAMemArray(),
                                    op.getBMemArray());
    rewriter.eraseOp(op);
    return success();
  }
};

// Pebble im2col: square HxW, per-cin plane, K=k^2.
constexpr int64_t kMaxIter = 34;
constexpr int64_t kMaxK = 7;
constexpr int64_t kBankLines = 1024;
constexpr int64_t kLane = 16;

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static int64_t pickTile(int64_t outDim, int64_t ksize, int64_t stride) {
  int64_t kElems = ksize * ksize;
  for (int64_t t = outDim; t >= 1; --t) {
    int64_t inSize = (t - 1) * stride + ksize;
    if (inSize > kMaxIter)
      continue;
    int64_t rows = cdiv(t * t, kLane) * cdiv(kElems, kLane) * kLane;
    if (rows <= kBankLines)
      return t;
  }
  return 0;
}

static int64_t pickFatCin(int64_t kElems, int64_t wins) {
  for (int64_t n = 256; n >= 1; --n) {
    int64_t kFat = n * kElems;
    if (kFat > 4095)
      continue;
    int64_t aRows = cdiv(wins, kLane) * cdiv(kFat, kLane) * kLane;
    if (aRows <= kBankLines)
      return n;
  }
  return 1;
}

static int64_t pickTileForConv(int64_t outDim, int64_t ksize, int64_t stride,
                               int64_t cin) {
  int64_t kElems = ksize * ksize;
  int64_t bestT = 0;
  int64_t bestScore = INT64_MAX;
  for (int64_t t = outDim; t >= 1; --t) {
    int64_t inSize = (t - 1) * stride + ksize;
    if (inSize > kMaxIter)
      continue;
    int64_t wins = t * t;
    int64_t rows = cdiv(wins, kLane) * cdiv(kElems, kLane) * kLane;
    if (rows > kBankLines)
      continue;
    int64_t fat = pickFatCin(kElems, wins);
    if (fat < 1)
      continue;
    int64_t spatial = cdiv(outDim, t) * cdiv(outDim, t);
    int64_t cinSteps = cdiv(cin, fat);
    int64_t score = spatial * cinSteps;
    if (score < bestScore) {
      bestScore = score;
      bestT = t;
    }
  }
  return bestT;
}

static void copyPlaneRows(OpBuilder &b, Location loc, Value plane, Value inPack,
                          int64_t lc, int64_t inRows, Value zero, Value one,
                          Value sixteen) {
  Value inRowsV = b.create<arith::ConstantIndexOp>(loc, inRows);
  Value base = b.create<arith::ConstantIndexOp>(loc, lc * inRows);
  auto rL = b.create<scf::ForOp>(loc, zero, inRowsV, one);
  b.setInsertionPointToStart(rL.getBody());
  Value r = rL.getInductionVar();
  Value rDst = b.create<arith::AddIOp>(loc, base, r);
  auto cL = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(cL.getBody());
  Value col = cL.getInductionVar();
  Value v = b.create<memref::LoadOp>(loc, plane, ValueRange{r, col});
  b.create<memref::StoreOp>(loc, v, inPack, ValueRange{rDst, col});
  b.setInsertionPointAfter(rL);
}

static void packInPlane(OpBuilder &b, Location loc, Value inPack, Value input,
                        Value nV, Value cV, Value ih0V, Value iw0V,
                        int64_t inSize, int64_t inRows, int64_t H, int64_t W,
                        int64_t padLow, Value zero, Value one, Value sixteen,
                        Value f0) {
  b.create<linalg::FillOp>(loc, f0, inPack);
  Value inSizeV = b.create<arith::ConstantIndexOp>(loc, inSize);
  Value nElems = b.create<arith::ConstantIndexOp>(loc, inSize * inSize);
  Value hV = b.create<arith::ConstantIndexOp>(loc, H);
  Value wV = b.create<arith::ConstantIndexOp>(loc, W);
  Value padV = b.create<arith::ConstantIndexOp>(loc, padLow);
  auto rL = b.create<scf::ForOp>(
      loc, zero, b.create<arith::ConstantIndexOp>(loc, inRows), one);
  b.setInsertionPointToStart(rL.getBody());
  auto cL = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(cL.getBody());
  Value flat = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, rL.getInductionVar(), sixteen),
      cL.getInductionVar());
  Value inBound =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, flat, nElems);
  Value flatC = b.create<arith::SelectOp>(loc, inBound, flat, zero);
  Value ih = b.create<arith::DivUIOp>(loc, flatC, inSizeV);
  Value iw = b.create<arith::RemUIOp>(loc, flatC, inSizeV);
  Value ihAbs = b.create<arith::AddIOp>(loc, ih0V, ih);
  Value iwAbs = b.create<arith::AddIOp>(loc, iw0V, iw);
  Value ihReal = b.create<arith::SubIOp>(loc, ihAbs, padV);
  Value iwReal = b.create<arith::SubIOp>(loc, iwAbs, padV);
  Value ihGe0 =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ihReal, zero);
  Value iwGe0 =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iwReal, zero);
  Value ihLt =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ihReal, hV);
  Value iwLt =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iwReal, wV);
  Value inImg =
      b.create<arith::AndIOp>(loc, b.create<arith::AndIOp>(loc, ihGe0, iwGe0),
                              b.create<arith::AndIOp>(loc, ihLt, iwLt));
  Value ok = b.create<arith::AndIOp>(loc, inBound, inImg);
  Value ihSafe = b.create<arith::SelectOp>(loc, ok, ihReal, zero);
  Value iwSafe = b.create<arith::SelectOp>(loc, ok, iwReal, zero);
  SmallVector<Value, 4> inIdx{nV, ihSafe, iwSafe, cV};
  Value v = b.create<memref::LoadOp>(loc, input, inIdx);
  v = b.create<arith::SelectOp>(loc, ok, v, f0);
  b.create<memref::StoreOp>(
      loc, v, inPack, ValueRange{rL.getInductionVar(), cL.getInductionVar()});
  b.setInsertionPointAfter(rL);
}

static void packFatFilter(OpBuilder &b, Location loc, Value filter, Value fPack,
                          Value c0V, int64_t nCin, int64_t KH, int64_t KW,
                          int64_t bRows, Value ocEnd, Value zero, Value one) {
  for (int64_t lc = 0; lc < nCin; ++lc) {
    Value lcV = b.create<arith::ConstantIndexOp>(loc, lc);
    Value cV = b.create<arith::AddIOp>(loc, c0V, lcV);
    for (int64_t kr = 0; kr < KH; ++kr) {
      for (int64_t kc = 0; kc < KW; ++kc) {
        int64_t k = kr * KH + kc;
        Value krV = b.create<arith::ConstantIndexOp>(loc, kr);
        Value kcV = b.create<arith::ConstantIndexOp>(loc, kc);
        Value rowV = b.create<arith::ConstantIndexOp>(loc, lc * bRows + k);
        auto oL = b.create<scf::ForOp>(loc, zero, ocEnd, one);
        b.setInsertionPointToStart(oL.getBody());
        Value oV = oL.getInductionVar();
        Value wt =
            b.create<memref::LoadOp>(loc, filter, ValueRange{krV, kcV, cV, oV});
        b.create<memref::StoreOp>(loc, wt, fPack, ValueRange{rowV, oV});
        b.setInsertionPointAfter(oL);
      }
    }
  }
}

static void packDwFilter(OpBuilder &b, Location loc, Value filter, Value fPack,
                         Value cV, int64_t lc, int64_t KH, int64_t KW,
                         int64_t bRows) {
  for (int64_t kr = 0; kr < KH; ++kr) {
    for (int64_t kc = 0; kc < KW; ++kc) {
      int64_t k = kr * KH + kc;
      Value krV = b.create<arith::ConstantIndexOp>(loc, kr);
      Value kcV = b.create<arith::ConstantIndexOp>(loc, kc);
      Value rowV = b.create<arith::ConstantIndexOp>(loc, lc * bRows + k);
      Value colV = b.create<arith::ConstantIndexOp>(loc, lc);
      Value zeroC = b.create<arith::ConstantIndexOp>(loc, 0);
      Value wt = b.create<memref::LoadOp>(loc, filter,
                                          ValueRange{krV, kcV, cV, zeroC});
      b.create<memref::StoreOp>(loc, wt, fPack, ValueRange{rowV, colV});
    }
  }
}

static void scatterOut(OpBuilder &b, Location loc, Value src, Value output,
                       Value nV, Value oh0, Value ow0, Value c0V, Value nChV,
                       Value tileV, Value ohLim, Value owLim, Value zero,
                       Value one) {
  auto ohL = b.create<scf::ForOp>(loc, zero, tileV, one);
  b.setInsertionPointToStart(ohL.getBody());
  auto owL = b.create<scf::ForOp>(loc, zero, tileV, one);
  b.setInsertionPointToStart(owL.getBody());
  Value loh = ohL.getInductionVar();
  Value low = owL.getInductionVar();
  Value oh = b.create<arith::AddIOp>(loc, oh0, loh);
  Value ow = b.create<arith::AddIOp>(loc, ow0, low);
  Value win = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, loh, tileV), low);
  Value ohOk =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oh, ohLim);
  Value owOk =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, ow, owLim);
  Value ok = b.create<arith::AndIOp>(loc, ohOk, owOk);
  auto ifOp = b.create<scf::IfOp>(loc, ok, /*withElseRegion=*/false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto cL = b.create<scf::ForOp>(loc, zero, nChV, one);
  b.setInsertionPointToStart(cL.getBody());
  Value ov =
      b.create<memref::LoadOp>(loc, src, ValueRange{win, cL.getInductionVar()});
  Value cGlobal = b.create<arith::AddIOp>(loc, c0V, cL.getInductionVar());
  SmallVector<Value, 4> outIdx{nV, oh, ow, cGlobal};
  Value cur = b.create<memref::LoadOp>(loc, output, outIdx);
  b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, cur, ov), output,
                            outIdx);
  b.setInsertionPointAfter(ohL);
}

static void scatterMap(OpBuilder &b, Location loc, Value src, Value output,
                       Value nV, Value c0V, Value nChV, Value ohLim,
                       Value owLim, Value zero, Value one) {
  auto ohL = b.create<scf::ForOp>(loc, zero, ohLim, one);
  b.setInsertionPointToStart(ohL.getBody());
  auto owL = b.create<scf::ForOp>(loc, zero, owLim, one);
  b.setInsertionPointToStart(owL.getBody());
  Value oh = ohL.getInductionVar();
  Value ow = owL.getInductionVar();
  Value win =
      b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, oh, owLim), ow);
  auto cL = b.create<scf::ForOp>(loc, zero, nChV, one);
  b.setInsertionPointToStart(cL.getBody());
  Value ov =
      b.create<memref::LoadOp>(loc, src, ValueRange{win, cL.getInductionVar()});
  Value cGlobal = b.create<arith::AddIOp>(loc, c0V, cL.getInductionVar());
  SmallVector<Value, 4> outIdx{nV, oh, ow, cGlobal};
  Value cur = b.create<memref::LoadOp>(loc, output, outIdx);
  b.create<memref::StoreOp>(loc, b.create<arith::AddFOp>(loc, cur, ov), output,
                            outIdx);
  b.setInsertionPointAfter(ohL);
}

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileConv2dOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return op.emitError("requires static memrefs");

    Type elem = inTy.getElementType();
    if (isa<FloatType>(elem) && fTy.getElementType() == elem &&
        oTy.getElementType() == elem) {
      auto is = inTy.getShape(), fs = fTy.getShape(), os = oTy.getShape();
      int64_t H = is[1], W = is[2];
      int64_t KH = fs[0], KW = fs[1];
      int64_t OH = os[1], OW = os[2];
      int64_t padLow = 0, padHigh = 0;
      if (auto a = op->getAttrOfType<IntegerAttr>("pad_low"))
        padLow = a.getInt();
      if (auto a = op->getAttrOfType<IntegerAttr>("pad_high"))
        padHigh = a.getInt();
      if (padLow != 0 || padHigh != 0)
        return op.emitError("float compact conv path requires pad=0");
      if (H != W || OH != OW || KH != KW)
        return op.emitError("float compact conv requires square H/W/K");
      int64_t stride;
      if (OH == 1)
        stride = H - KH + 1;
      else {
        if ((H - KH) % (OH - 1) != 0)
          return op.emitError("cannot infer integer stride from shapes");
        stride = (H - KH) / (OH - 1);
      }
      if (stride < 1 || (H - KH) / stride + 1 != OH)
        return op.emitError("pad/stride/shape mismatch");
      auto ones = b.getI64TensorAttr({stride, stride});
      auto dilations = b.getI64TensorAttr({1, 1});
      b.create<linalg::Conv2DNhwcHwcfOp>(
          loc, TypeRange{}, ValueRange{op.getInput(), op.getFilter()},
          ValueRange{op.getOutput()}, ones, dilations);
      b.eraseOp(op);
      return success();
    }

    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return op.emitError(
          "requires static FP32 activation, INT8 filter, FP32 output");

    auto dwAddrAttr = op->getAttrOfType<IntegerAttr>("dw_addr");
    auto dwBytesAttr = op->getAttrOfType<IntegerAttr>("dw_bytes");
    auto perChannelAttr = op->getAttrOfType<BoolAttr>("per_channel");
    if (!dwAddrAttr || !dwBytesAttr || !perChannelAttr ||
        dwAddrAttr.getInt() < 16)
      return op.emitError("quantized conv requires RAX Dw metadata");

    auto is = inTy.getShape(), fs = fTy.getShape(), os = oTy.getShape();
    int64_t N = is[0], H = is[1], W = is[2], C = is[3];
    int64_t KH = fs[0], KW = fs[1], FC = fs[2], OC = fs[3];
    int64_t OH = os[1], OW = os[2];
    int64_t padLow = 0, padHigh = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("pad_low"))
      padLow = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("pad_high"))
      padHigh = a.getInt();
    if (N != os[0] || C != FC || OC != os[3] || H != W || OH != OW || KH != KW)
      return op.emitError("shape mismatch or non-square H/W/K");
    const int64_t ocPadForDw = cdiv(OC, kLane) * kLane;
    const int64_t dwRequired = perChannelAttr.getValue() ? ocPadForDw * 4 : 4;
    if (dwBytesAttr.getInt() < dwRequired)
      return op.emitError(
          "Dw scale image does not cover padded output channels");
    if (KH < 1 || KH > kMaxK)
      return op.emitError("ksize out of range");
    if (padLow < 0 || padHigh < 0 || padLow > 7 || padHigh > 7)
      return op.emitError("pad out of range");
    int64_t padded = H + padLow + padHigh;
    if (padded < KH)
      return op.emitError("padded input smaller than kernel");
    int64_t stride;
    if (OH == 1)
      stride = padded - KH + 1;
    else {
      if ((padded - KH) % (OH - 1) != 0)
        return op.emitError("cannot infer integer stride from shapes");
      stride = (padded - KH) / (OH - 1);
    }
    if (stride < 1 || (padded - KH) / stride + 1 != OH)
      return op.emitError("pad/stride/shape mismatch");

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value f0 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(0.0f));
    Value i0 = b.create<arith::ConstantOp>(loc, fTy.getElementType(),
                                           b.getZeroAttr(fTy.getElementType()));

    if (KH == 1) {
      if (padLow != 0 || padHigh != 0)
        return op.emitError("1x1 conv requires pad=0");
      if ((OH - 1) * stride + 1 != H || (OW - 1) * stride + 1 != W)
        return op.emitError("1x1 conv stride/shape mismatch");
      int64_t M = OH * OW;
      int64_t mPad = cdiv(M, kLane) * kLane;
      int64_t kPad = cdiv(C, kLane) * kLane;
      Value ohEnd = b.create<arith::ConstantIndexOp>(loc, OH);
      Value owEnd = b.create<arith::ConstantIndexOp>(loc, OW);
      Value cEnd = b.create<arith::ConstantIndexOp>(loc, C);
      Value strideV = b.create<arith::ConstantIndexOp>(loc, stride);
      Value aPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({mPad, kPad}, b.getF32Type()));
      Value bPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({kPad, kLane}, fTy.getElementType()));
      Value cPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({mPad, kLane}, b.getF32Type()));

      for (int64_t n = 0; n < N; ++n) {
        Value nV = b.create<arith::ConstantIndexOp>(loc, n);
        b.create<linalg::FillOp>(loc, f0, aPack);
        auto ohL = b.create<scf::ForOp>(loc, zero, ohEnd, one);
        b.setInsertionPointToStart(ohL.getBody());
        auto owL = b.create<scf::ForOp>(loc, zero, owEnd, one);
        b.setInsertionPointToStart(owL.getBody());
        Value oh = ohL.getInductionVar();
        Value ow = owL.getInductionVar();
        Value ih = b.create<arith::MulIOp>(loc, oh, strideV);
        Value iw = b.create<arith::MulIOp>(loc, ow, strideV);
        Value win = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, oh, owEnd), ow);
        auto cL = b.create<scf::ForOp>(loc, zero, cEnd, one);
        b.setInsertionPointToStart(cL.getBody());
        Value cV = cL.getInductionVar();
        Value v = b.create<memref::LoadOp>(loc, op.getInput(),
                                           ValueRange{nV, ih, iw, cV});
        b.create<memref::StoreOp>(loc, v, aPack, ValueRange{win, cV});
        b.setInsertionPointAfter(ohL);

        for (int64_t oc0 = 0; oc0 < OC; oc0 += kLane) {
          int64_t ocTile = OC - oc0 < kLane ? OC - oc0 : kLane;
          Value oc0V = b.create<arith::ConstantIndexOp>(loc, oc0);
          Value ocTileV = b.create<arith::ConstantIndexOp>(loc, ocTile);
          const int64_t panelDwOffset = perChannelAttr.getValue() ? oc0 * 4 : 0;
          auto panelDwAddr =
              b.getI64IntegerAttr(dwAddrAttr.getInt() + panelDwOffset);
          auto panelDwBytes =
              b.getI64IntegerAttr(dwBytesAttr.getInt() - panelDwOffset);
          b.create<linalg::FillOp>(loc, i0, bPack);
          auto cPackL = b.create<scf::ForOp>(loc, zero, cEnd, one);
          b.setInsertionPointToStart(cPackL.getBody());
          Value cPv = cPackL.getInductionVar();
          auto oPackL = b.create<scf::ForOp>(loc, zero, ocTileV, one);
          b.setInsertionPointToStart(oPackL.getBody());
          Value oLocal = oPackL.getInductionVar();
          Value oGlobal = b.create<arith::AddIOp>(loc, oc0V, oLocal);
          Value wt = b.create<memref::LoadOp>(
              loc, op.getFilter(), ValueRange{zero, zero, cPv, oGlobal});
          b.create<memref::StoreOp>(loc, wt, bPack, ValueRange{cPv, oLocal});
          b.setInsertionPointAfter(cPackL);

          if (mPad <= kBankLines) {
            b.create<linalg::FillOp>(loc, f0, cPack);
            auto matmul = b.create<SMatMulMatmulOp>(loc, aPack, bPack, cPack);
            matmul->setAttr("dwAddr", panelDwAddr);
            matmul->setAttr("dwBytes", panelDwBytes);
            matmul->setAttr("perChannel", perChannelAttr);
          } else {
            int64_t aligned = (mPad / kBankLines) * kBankLines;
            int64_t rem = mPad - aligned;
            Value step = b.create<arith::ConstantIndexOp>(loc, kBankLines);
            Value alignedV = b.create<arith::ConstantIndexOp>(loc, aligned);
            auto mL = b.create<scf::ForOp>(loc, zero, alignedV, step);
            b.setInsertionPointToStart(mL.getBody());
            Value aTile = b.create<memref::SubViewOp>(
                loc, aPack,
                SmallVector<OpFoldResult>{mL.getInductionVar(),
                                          b.getIndexAttr(0)},
                SmallVector<OpFoldResult>{b.getIndexAttr(kBankLines),
                                          b.getIndexAttr(kPad)},
                SmallVector<OpFoldResult>{b.getIndexAttr(1),
                                          b.getIndexAttr(1)});
            Value cTile = b.create<memref::SubViewOp>(
                loc, cPack,
                SmallVector<OpFoldResult>{mL.getInductionVar(),
                                          b.getIndexAttr(0)},
                SmallVector<OpFoldResult>{b.getIndexAttr(kBankLines),
                                          b.getIndexAttr(kLane)},
                SmallVector<OpFoldResult>{b.getIndexAttr(1),
                                          b.getIndexAttr(1)});
            auto matmul = b.create<SMatMulMatmulOp>(loc, aTile, bPack, cTile);
            matmul->setAttr("dwAddr", panelDwAddr);
            matmul->setAttr("dwBytes", panelDwBytes);
            matmul->setAttr("perChannel", perChannelAttr);
            b.setInsertionPointAfter(mL);
            if (rem > 0) {
              Value aRem = b.create<memref::SubViewOp>(
                  loc, aPack,
                  SmallVector<OpFoldResult>{b.getIndexAttr(aligned),
                                            b.getIndexAttr(0)},
                  SmallVector<OpFoldResult>{b.getIndexAttr(rem),
                                            b.getIndexAttr(kPad)},
                  SmallVector<OpFoldResult>{b.getIndexAttr(1),
                                            b.getIndexAttr(1)});
              Value cRem = b.create<memref::SubViewOp>(
                  loc, cPack,
                  SmallVector<OpFoldResult>{b.getIndexAttr(aligned),
                                            b.getIndexAttr(0)},
                  SmallVector<OpFoldResult>{b.getIndexAttr(rem),
                                            b.getIndexAttr(kLane)},
                  SmallVector<OpFoldResult>{b.getIndexAttr(1),
                                            b.getIndexAttr(1)});
              auto matmul = b.create<SMatMulMatmulOp>(loc, aRem, bPack, cRem);
              matmul->setAttr("dwAddr", panelDwAddr);
              matmul->setAttr("dwBytes", panelDwBytes);
              matmul->setAttr("perChannel", perChannelAttr);
            }
          }
          scatterMap(b, loc, cPack, op.getOutput(), nV, oc0V, ocTileV, ohEnd,
                     owEnd, zero, one);
        }
      }

      b.create<memref::DeallocOp>(loc, aPack);
      b.create<memref::DeallocOp>(loc, bPack);
      b.create<memref::DeallocOp>(loc, cPack);
      b.eraseOp(op);
      return success();
    }

    int64_t tile = pickTileForConv(OH, KH, stride, C);
    if (tile == 0)
      return op.emitError("no tile fits im2col bank capacity");

    int64_t kElems = KH * KH;
    int64_t wins = tile * tile;
    int64_t inSize = (tile - 1) * stride + KH;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t bRows = cdiv(kElems, kLane) * kLane;
    if (wins > kBankLines)
      return op.emitError("conv M pad exceeds bank depth");

    Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
    Value tileV = b.create<arith::ConstantIndexOp>(loc, tile);
    Value strideV = b.create<arith::ConstantIndexOp>(loc, stride);
    Value ohEnd = b.create<arith::ConstantIndexOp>(loc, OH);
    Value owEnd = b.create<arith::ConstantIndexOp>(loc, OW);

    int64_t ocPad = cdiv(OC, kLane) * kLane;
    if (ocPad < kLane || ocPad > 4096)
      return op.emitError("OC pad out of range");

    int64_t fatCinMax = pickFatCin(kElems, wins);
    if (fatCinMax < 1)
      return op.emitError("no fat cin batch fits im2col bank capacity");

    Value inPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({fatCinMax * inRows, kLane}, b.getF32Type()));
    Value fPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({fatCinMax * bRows, ocPad}, fTy.getElementType()));
    Value tmpF = b.create<memref::AllocOp>(
        loc, MemRefType::get({wins, ocPad}, b.getF32Type()));
    Value accF = b.create<memref::AllocOp>(
        loc, MemRefType::get({wins, ocPad}, b.getF32Type()));
    Value plane = b.create<memref::AllocOp>(
        loc, MemRefType::get({inRows, kLane}, b.getF32Type()));
    Value ocPadV = b.create<arith::ConstantIndexOp>(loc, ocPad);
    Value ocEnd = b.create<arith::ConstantIndexOp>(loc, OC);
    Value oc0V = zero;

    for (int64_t n = 0; n < N; ++n) {
      Value nV = b.create<arith::ConstantIndexOp>(loc, n);
      auto oh0L = b.create<scf::ForOp>(loc, zero, ohEnd, tileV);
      b.setInsertionPointToStart(oh0L.getBody());
      Value oh0 = oh0L.getInductionVar();
      auto ow0L = b.create<scf::ForOp>(loc, zero, owEnd, tileV);
      b.setInsertionPointToStart(ow0L.getBody());
      Value ow0 = ow0L.getInductionVar();
      Value ih0V = b.create<arith::MulIOp>(loc, oh0, strideV);
      Value iw0V = b.create<arith::MulIOp>(loc, ow0, strideV);

      b.create<linalg::FillOp>(loc, f0, accF);
      for (int64_t c0 = 0; c0 < C; c0 += fatCinMax) {
        int64_t nCin = C - c0 < fatCinMax ? C - c0 : fatCinMax;
        Value c0V = b.create<arith::ConstantIndexOp>(loc, c0);
        b.create<linalg::FillOp>(loc, i0, fPack);
        for (int64_t lc = 0; lc < nCin; ++lc) {
          Value cV = b.create<arith::ConstantIndexOp>(loc, c0 + lc);
          packInPlane(b, loc, plane, op.getInput(), nV, cV, ih0V, iw0V, inSize,
                      inRows, H, W, padLow, zero, one, sixteen, f0);
          copyPlaneRows(b, loc, plane, inPack, lc, inRows, zero, one, sixteen);
        }
        packFatFilter(b, loc, op.getFilter(), fPack, c0V, nCin, KH, KW, bRows,
                      ocEnd, zero, one);
        Value inSub = b.create<memref::SubViewOp>(
            loc, inPack,
            ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(nCin * inRows),
                                   b.getIndexAttr(kLane)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        Value fSub = b.create<memref::SubViewOp>(
            loc, fPack,
            ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(nCin * bRows),
                                   b.getIndexAttr(ocPad)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        auto matmul = b.create<Im2colFatMatmulOp>(
            loc, inSub, fSub, tmpF, b.getI64IntegerAttr(inSize),
            b.getI64IntegerAttr(KH), b.getI64IntegerAttr(ocPad),
            b.getI64IntegerAttr(nCin), b.getI64IntegerAttr(stride),
            b.getI64IntegerAttr(0));
        matmul->setAttr("dwAddr", dwAddrAttr);
        matmul->setAttr("dwBytes", dwBytesAttr);
        matmul->setAttr("perChannel", perChannelAttr);
        Value zI = b.create<arith::ConstantIndexOp>(loc, 0);
        Value nRows = b.create<arith::ConstantIndexOp>(loc, wins);
        auto rAdd = b.create<scf::ForOp>(loc, zI, nRows, one);
        b.setInsertionPointToStart(rAdd.getBody());
        auto cAdd = b.create<scf::ForOp>(loc, zI, ocPadV, one);
        b.setInsertionPointToStart(cAdd.getBody());
        Value aa = b.create<memref::LoadOp>(
            loc, accF,
            ValueRange{rAdd.getInductionVar(), cAdd.getInductionVar()});
        Value tt = b.create<memref::LoadOp>(
            loc, tmpF,
            ValueRange{rAdd.getInductionVar(), cAdd.getInductionVar()});
        b.create<memref::StoreOp>(
            loc, b.create<arith::AddFOp>(loc, aa, tt), accF,
            ValueRange{rAdd.getInductionVar(), cAdd.getInductionVar()});
        b.setInsertionPointAfter(rAdd);
      }

      scatterOut(b, loc, accF, op.getOutput(), nV, oh0, ow0, oc0V, ocEnd, tileV,
                 ohEnd, owEnd, zero, one);
      b.setInsertionPointAfter(oh0L);
    }

    b.create<memref::DeallocOp>(loc, inPack);
    b.create<memref::DeallocOp>(loc, fPack);
    b.create<memref::DeallocOp>(loc, tmpF);
    b.create<memref::DeallocOp>(loc, accF);
    b.create<memref::DeallocOp>(loc, plane);
    b.eraseOp(op);
    return success();
  }
};

// Depthwise: channel tiles of 16 -> one fat im2col+GEMM (block-diag B), never
// GEMV.
class TileDepthwiseConv2dLowering
    : public OpRewritePattern<tile::TileDepthwiseConv2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileDepthwiseConv2dOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape() ||
        !inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return op.emitError(
          "requires static FP32 activations, INT8 weights, and FP32 output");

    auto dwAddr = op->getAttrOfType<IntegerAttr>("dw_addr");
    auto dwBytes = op->getAttrOfType<IntegerAttr>("dw_bytes");
    auto perChannel = op->getAttrOfType<BoolAttr>("per_channel");
    if (!dwAddr || !dwBytes || !perChannel)
      return op.emitError("requires Dw RAX metadata");

    auto is = inTy.getShape(), fs = fTy.getShape(), os = oTy.getShape();
    int64_t N = is[0], H = is[1], W = is[2], C = is[3];
    int64_t KH = fs[0], KW = fs[1], FC = fs[2], mult = fs[3];
    int64_t OH = os[1], OW = os[2], OC = os[3];
    int64_t padLow = 0, padHigh = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("pad_low"))
      padLow = a.getInt();
    if (auto a = op->getAttrOfType<IntegerAttr>("pad_high"))
      padHigh = a.getInt();
    if (N != os[0] || C != FC || OC != C || mult != 1 || H != W || OH != OW ||
        KH != KW)
      return op.emitError("depthwise shape mismatch");
    const int64_t cPadForDw = cdiv(C, kLane) * kLane;
    const int64_t dwRequired = perChannel.getValue() ? cPadForDw * 4 : 4;
    if (dwBytes.getInt() < dwRequired)
      return op.emitError(
          "Dw scale image does not cover padded output channels");
    if (KH < 1 || KH > kMaxK)
      return op.emitError("ksize out of range");
    if (padLow < 0 || padHigh < 0 || padLow > 7 || padHigh > 7)
      return op.emitError("pad out of range");
    int64_t padded = H + padLow + padHigh;
    if (padded < KH)
      return op.emitError("padded input smaller than kernel");
    int64_t stride;
    if (OH == 1)
      stride = padded - KH + 1;
    else {
      if ((padded - KH) % (OH - 1) != 0)
        return op.emitError("cannot infer integer stride from shapes");
      stride = (padded - KH) / (OH - 1);
    }
    if (stride < 1 || (padded - KH) / stride + 1 != OH)
      return op.emitError("pad/stride/shape mismatch");

    // Spatial tile by single-plane im2col capacity; channels batched by lane.
    int64_t tile = pickTile(OH, KH, stride);
    int64_t nBatch = kLane;
    if (tile == 0)
      return op.emitError("no depthwise tile fits im2col bank capacity");

    int64_t kElems = KH * KH;
    int64_t wins = tile * tile;
    int64_t inSize = (tile - 1) * stride + KH;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t bRows = cdiv(kElems, kLane) * kLane;

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
    Value tileV = b.create<arith::ConstantIndexOp>(loc, tile);
    Value strideV = b.create<arith::ConstantIndexOp>(loc, stride);
    Value ohEnd = b.create<arith::ConstantIndexOp>(loc, OH);
    Value owEnd = b.create<arith::ConstantIndexOp>(loc, OW);
    Value f0 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));

    Value inPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({nBatch * inRows, kLane}, b.getF32Type()));
    Value fPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({nBatch * bRows, kLane}, b.getI8Type()));
    Value outF = b.create<memref::AllocOp>(
        loc, MemRefType::get({wins, kLane}, b.getF32Type()));
    Value plane = b.create<memref::AllocOp>(
        loc, MemRefType::get({inRows, kLane}, b.getF32Type()));

    for (int64_t n = 0; n < N; ++n) {
      Value nV = b.create<arith::ConstantIndexOp>(loc, n);
      auto oh0L = b.create<scf::ForOp>(loc, zero, ohEnd, tileV);
      b.setInsertionPointToStart(oh0L.getBody());
      Value oh0 = oh0L.getInductionVar();
      auto ow0L = b.create<scf::ForOp>(loc, zero, owEnd, tileV);
      b.setInsertionPointToStart(ow0L.getBody());
      Value ow0 = ow0L.getInductionVar();
      Value ih0V = b.create<arith::MulIOp>(loc, oh0, strideV);
      Value iw0V = b.create<arith::MulIOp>(loc, ow0, strideV);

      for (int64_t c0 = 0; c0 < C; c0 += nBatch) {
        int64_t nCh = C - c0 < nBatch ? C - c0 : nBatch;
        b.create<linalg::FillOp>(loc, f0, inPack);
        b.create<linalg::FillOp>(loc, f0, fPack);

        for (int64_t lc = 0; lc < nCh; ++lc) {
          Value cV = b.create<arith::ConstantIndexOp>(loc, c0 + lc);
          packInPlane(b, loc, plane, op.getInput(), nV, cV, ih0V, iw0V, inSize,
                      inRows, H, W, padLow, zero, one, sixteen, f0);
          copyPlaneRows(b, loc, plane, inPack, lc, inRows, zero, one, sixteen);
          packDwFilter(b, loc, op.getFilter(), fPack, cV, lc, KH, KW, bRows);
        }

        auto matmul = b.create<Im2colDepthwiseMatmulOp>(
            loc, inPack, fPack, outF, b.getI64IntegerAttr(inSize),
            b.getI64IntegerAttr(KH), b.getI64IntegerAttr(nBatch),
            b.getI64IntegerAttr(stride), b.getI64IntegerAttr(0));
        const int64_t panelDwOffset = perChannel.getValue() ? c0 * 4 : 0;
        matmul->setAttr("dwAddr",
                        b.getI64IntegerAttr(dwAddr.getInt() + panelDwOffset));
        matmul->setAttr("dwBytes",
                        b.getI64IntegerAttr(dwBytes.getInt() - panelDwOffset));
        matmul->setAttr("perChannel", perChannel);

        Value nChV = b.create<arith::ConstantIndexOp>(loc, nCh);
        Value c0V = b.create<arith::ConstantIndexOp>(loc, c0);
        scatterOut(b, loc, outF, op.getOutput(), nV, oh0, ow0, c0V, nChV, tileV,
                   ohEnd, owEnd, zero, one);
      }
      b.setInsertionPointAfter(oh0L);
    }

    b.create<memref::DeallocOp>(loc, inPack);
    b.create<memref::DeallocOp>(loc, fPack);
    b.create<memref::DeallocOp>(loc, outF);
    b.create<memref::DeallocOp>(loc, plane);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  populateSMatMulBallTileLoweringPatterns(patterns, bankWidthBytes, bankDepth,
                                           bankNum);
  patterns.add<TileTransposeLowering>(patterns.getContext());
  patterns.add<TileConv2dLowering>(patterns.getContext());
  patterns.add<TileDepthwiseConv2dLowering>(patterns.getContext());
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
                            llvm::cl::init(1024)};
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
    target.addIllegalOp<::buddy::tile::TileMatMulOp>();
    target.addIllegalOp<::buddy::tile::TileConv2dOp>();
    target.addIllegalOp<::buddy::tile::TileDepthwiseConv2dOp>();
    target.addIllegalOp<::buddy::tile::TileTransposeOp>();

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
