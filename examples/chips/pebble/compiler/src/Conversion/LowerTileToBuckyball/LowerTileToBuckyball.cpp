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
#include "Utils/BankUtils.h"
#include "Utils/QuantUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;
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

// Bank mvin/mvout stride is (memref row stride in elems) / 16.
static LogicalResult rowStrideDiv16(MemRefType ty, int64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % 16 != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = strides[0] / 16;
  return success();
}

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  TileTransposeLowering(MLIRContext *context, int64_t bankWidthBytes)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes) {}

  LogicalResult matchAndRewrite(tile::TileTransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getAMemArray();
    Value output = op.getBMemArray();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return op.emitError("requires static input and output memrefs");

    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();
    size_t rows = inShape[0];
    size_t cols = inShape[1];
    if (outShape[0] != static_cast<int64_t>(cols) ||
        outShape[1] != static_cast<int64_t>(rows))
      return op.emitError("output shape must transpose the input shape");

    Type elemTy = inputType.getElementType();
    int64_t elemBits = elemTy.getIntOrFloatBitWidth();
    if (elemBits != 8 && elemBits != 32)
      return op.emitError("only 8/32-bit transpose elements are supported");

    size_t elemsPerRow = elemsPerBankRow(elemTy, bankWidthBytes);
    if (elemsPerRow == 0)
      return op.emitError("unsupported transpose element type");

    // src+dst share pebble's 8 banks => at most 4 column-groups per side
    size_t colTile = elemsPerRow * 4;
    if (colTile == 0)
      return op.emitError("tile width is smaller than one bank row");

    // Transpose ball packs with iter==rowTile and W==colTile; pad to full
    // tiles. Also materialize when layout isn't static row-major (MobileNet
    // bufferization often yields strided<[?,?]>).
    size_t rowsPad = ceilDiv(rows, kMatmulTile) * kMatmulTile;
    size_t colsPad = ceilDiv(cols, colTile) * colTile;
    bool pad = rowsPad != rows || colsPad != cols;
    int64_t dummyStride = 0;
    bool inContig = succeeded(rowStrideDiv16(inputType, dummyStride));
    bool outContig = succeeded(rowStrideDiv16(outputType, dummyStride));
    // HW tiles need contiguous scratch on both sides.
    bool materialize = pad || !inContig || !outContig;

    Value inBuf = input;
    Value outBuf = output;
    if (materialize) {
      auto inPadTy =
          MemRefType::get({(int64_t)rowsPad, (int64_t)colsPad}, elemTy);
      auto outPadTy =
          MemRefType::get({(int64_t)colsPad, (int64_t)rowsPad}, elemTy);
      inBuf = rewriter.create<memref::AllocOp>(loc, inPadTy);
      outBuf = rewriter.create<memref::AllocOp>(loc, outPadTy);
      Value z = rewriter.create<arith::ConstantOp>(
          loc, elemTy, rewriter.getZeroAttr(elemTy));
      rewriter.create<linalg::FillOp>(loc, z, inBuf);
      rewriter.create<linalg::FillOp>(loc, z, outBuf);
      Value inView = rewriter.create<memref::SubViewOp>(
          loc, inBuf,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(rows),
                                    rewriter.getIndexAttr(cols)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, input, inView);
    }

    int64_t groups = (int64_t)(colTile / elemsPerRow);
    // rowsPad/colsPad equal the live shape when !materialize.
    int64_t strideIn = (int64_t)colsPad / 16;
    int64_t strideOut = (int64_t)rowsPad / 16;
    for (size_t r0 = 0; r0 < rowsPad; r0 += kMatmulTile) {
      for (size_t c0 = 0; c0 < colsPad; c0 += colTile) {
        Value inTile = rewriter.create<memref::SubViewOp>(
            loc, inBuf,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(r0),
                                      rewriter.getIndexAttr(c0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(kMatmulTile),
                                      rewriter.getIndexAttr(colTile)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, outBuf,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(c0),
                                      rewriter.getIndexAttr(r0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(colTile),
                                      rewriter.getIndexAttr(kMatmulTile)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        Value src = allocBank(rewriter, loc, 1, groups);
        Value dst = allocBank(rewriter, loc, 1, groups);
        Value loaded =
            mvinBank(rewriter, loc, inTile, src, kMatmulTile, strideIn);
        Value transposed = rewriter.create<BankTransposeOp>(
            loc, dst.getType(), loaded, dst,
            createI64Const(rewriter, loc, (int64_t)kMatmulTile),
            createI64Const(rewriter, loc, elemBits));
        mvoutBank(rewriter, loc, outTile, transposed, kMatmulTile, strideOut);
        releaseBank(rewriter, loc, loaded);
        releaseBank(rewriter, loc, transposed);
      }
    }

    if (materialize) {
      Value outView = rewriter.create<memref::SubViewOp>(
          loc, outBuf,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(cols),
                                    rewriter.getIndexAttr(rows)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, outView, output);
      rewriter.create<memref::DeallocOp>(loc, inBuf);
      rewriter.create<memref::DeallocOp>(loc, outBuf);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t bankWidthBytes;
};

// Pebble im2col: square HxW, per-cin plane, K=k^2.
// Conv: tile / cin / oc16 -> pack -> fp2int -> im2col -> matrix -> int2fp;
// add into outs. pad=0 stride=1 only. OC tiled by 16.
constexpr int64_t kMaxIter = 34;
constexpr int64_t kMaxK = 7;
constexpr int64_t kBankLines = 1024;
constexpr int64_t kLane = 16;

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static int64_t pickTile(int64_t outDim, int64_t ksize) {
  int64_t kElems = ksize * ksize;
  for (int64_t t = outDim; t >= 1; --t) {
    if (outDim % t != 0)
      continue;
    int64_t inSize = t + ksize - 1;
    if (inSize > kMaxIter)
      continue;
    // Input plane is packed into 16-wide bank rows with zero pad; no need for
    // inSize^2 % 16 == 0 (needed for 1x1 and many MobileNet shapes).
    int64_t rows = cdiv(t * t, kLane) * cdiv(kElems, kLane) * kLane;
    if (rows <= kBankLines)
      return t;
  }
  return 0;
}

static void addF32Pack(OpBuilder &b, Location loc, Value dst, Value src,
                       int64_t rows) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nRows = b.create<arith::ConstantIndexOp>(loc, rows);
  Value nCols = b.create<arith::ConstantIndexOp>(loc, kLane);
  auto r = b.create<scf::ForOp>(loc, zero, nRows, one);
  b.setInsertionPointToStart(r.getBody());
  auto c = b.create<scf::ForOp>(loc, zero, nCols, one);
  b.setInsertionPointToStart(c.getBody());
  Value a = b.create<memref::LoadOp>(
      loc, dst, ValueRange{r.getInductionVar(), c.getInductionVar()});
  Value t = b.create<memref::LoadOp>(
      loc, src, ValueRange{r.getInductionVar(), c.getInductionVar()});
  b.create<memref::StoreOp>(
      loc, b.create<arith::AddFOp>(loc, a, t), dst,
      ValueRange{r.getInductionVar(), c.getInductionVar()});
  b.setInsertionPointAfter(r);
}

static void packInPlane(OpBuilder &b, Location loc, Value inPack, Value input,
                        Value nV, Value cV, Value ih0V, Value iw0V,
                        int64_t inSize, int64_t inRows, Value zero, Value one,
                        Value sixteen, Value f0) {
  b.create<linalg::FillOp>(loc, f0, inPack);
  Value inSizeV = b.create<arith::ConstantIndexOp>(loc, inSize);
  Value nElems = b.create<arith::ConstantIndexOp>(loc, inSize * inSize);
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
  SmallVector<Value, 4> inIdx{nV, ihAbs, iwAbs, cV};
  Value v = b.create<memref::LoadOp>(loc, input, inIdx);
  v = b.create<arith::SelectOp>(loc, inBound, v, f0);
  b.create<memref::StoreOp>(
      loc, v, inPack, ValueRange{rL.getInductionVar(), cL.getInductionVar()});
  b.setInsertionPointAfter(rL);
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
        !fTy.hasStaticShape() || !oTy.hasStaticShape() ||
        !inTy.getElementType().isF32() || !fTy.getElementType().isF32() ||
        !oTy.getElementType().isF32())
      return op.emitError("requires static f32 memrefs");

    auto is = inTy.getShape(), fs = fTy.getShape(), os = oTy.getShape();
    int64_t N = is[0], H = is[1], W = is[2], C = is[3];
    int64_t KH = fs[0], KW = fs[1], FC = fs[2], OC = fs[3];
    int64_t OH = os[1], OW = os[2];
    if (N != os[0] || C != FC || OC != os[3] || H != W || OH != OW || KH != KW)
      return op.emitError("shape mismatch or non-square H/W/K");
    if (KH < 1 || KH > kMaxK)
      return op.emitError("ksize out of range");
    if (OH != H - KH + 1)
      return op.emitError("only pad=0 stride=1 supported");

    int64_t tile = pickTile(OH, KH);
    if (tile == 0)
      return op.emitError("no tile fits im2col bank capacity");

    int64_t kElems = KH * KH;
    int64_t wins = tile * tile;
    int64_t inSize = tile + KH - 1;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t bRows = cdiv(kElems, kLane) * kLane;
    int64_t cBlocks = wins;

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sixteen = b.create<arith::ConstantIndexOp>(loc, kLane);
    Value f0 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(0.0f));
    Value cEnd = b.create<arith::ConstantIndexOp>(loc, C);

    Value inPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({inRows, kLane}, b.getF32Type()));
    Value fPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({bRows, kLane}, b.getF32Type()));
    Value tmpI = b.create<memref::AllocOp>(
        loc, MemRefType::get({cBlocks, kLane}, b.getI32Type()));
    Value tmpF = b.create<memref::AllocOp>(
        loc, MemRefType::get({cBlocks, kLane}, b.getF32Type()));
    Value accF = b.create<memref::AllocOp>(
        loc, MemRefType::get({cBlocks, kLane}, b.getF32Type()));

    for (int64_t n = 0; n < N; ++n) {
      Value nV = b.create<arith::ConstantIndexOp>(loc, n);
      for (int64_t oh0 = 0; oh0 < OH; oh0 += tile) {
        for (int64_t ow0 = 0; ow0 < OW; ow0 += tile) {
          for (int64_t oc0 = 0; oc0 < OC; oc0 += kLane) {
            int64_t ocTile = OC - oc0 < kLane ? OC - oc0 : kLane;
            uint64_t cfg = packBits(wins, 0, 11) | packBits(ocTile, 12, 23) |
                           packBits(kElems, 24, 35);
            b.create<linalg::FillOp>(loc, f0, accF);

            Value ih0V = b.create<arith::ConstantIndexOp>(loc, oh0);
            Value iw0V = b.create<arith::ConstantIndexOp>(loc, ow0);
            auto cinL = b.create<scf::ForOp>(loc, zero, cEnd, one);
            b.setInsertionPointToStart(cinL.getBody());
            Value cV = cinL.getInductionVar();

            packInPlane(b, loc, inPack, op.getInput(), nV, cV, ih0V, iw0V,
                        inSize, inRows, zero, one, sixteen, f0);

            b.create<linalg::FillOp>(loc, f0, fPack);
            for (int64_t kr = 0; kr < KH; ++kr) {
              for (int64_t kc = 0; kc < KW; ++kc) {
                int64_t k = kr * KH + kc;
                Value krV = b.create<arith::ConstantIndexOp>(loc, kr);
                Value kcV = b.create<arith::ConstantIndexOp>(loc, kc);
                Value rowV = b.create<arith::ConstantIndexOp>(loc, k);
                for (int64_t o = 0; o < ocTile; ++o) {
                  Value oLocal = b.create<arith::ConstantIndexOp>(loc, o);
                  Value oGlobal =
                      b.create<arith::ConstantIndexOp>(loc, oc0 + o);
                  Value wt = b.create<memref::LoadOp>(
                      loc, op.getFilter(), ValueRange{krV, kcV, cV, oGlobal});
                  b.create<memref::StoreOp>(loc, wt, fPack,
                                            ValueRange{rowV, oLocal});
                }
              }
            }

            Value scaleAF =
                quantScale(b, loc, absMaxF32(b, loc, inPack, inRows, kLane));
            Value scaleBF =
                quantScale(b, loc, absMaxF32(b, loc, fPack, bRows, kLane));
            Value scaleA = packF32BitsAsI64(b, loc, scaleAF);
            Value scaleB = packF32BitsAsI64(b, loc, scaleBF);
            Value scaleD = packF32BitsAsI64(
                b, loc, dequantScale(b, loc, scaleAF, scaleBF));

            Value inFB = allocBank(b, loc, 1, 4);
            Value inIB = allocBank(b, loc, 1, 1);
            Value loaded = mvinBank(b, loc, inPack, inFB, inRows);
            Value quant =
                b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                                       createI64Const(b, loc, inRows), scaleA);
            releaseBank(b, loc, loaded);

            Value patches = allocBank(b, loc, 1, 1);
            Value patch = b.create<BankIm2colOp>(
                loc, patches.getType(), quant, patches,
                createI64Const(b, loc, inSize), createI64Const(b, loc, KH),
                createI64Const(b, loc, 1), createI64Const(b, loc, 0));
            releaseBank(b, loc, quant);

            Value fFB = allocBank(b, loc, 1, 4);
            Value fIB = allocBank(b, loc, 1, 1);
            Value fLoaded = mvinBank(b, loc, fPack, fFB, bRows);
            Value fQuant =
                b.create<BankFp2IntOp>(loc, fIB.getType(), fLoaded, fIB,
                                       createI64Const(b, loc, bRows), scaleB);
            releaseBank(b, loc, fLoaded);

            Value accB = allocBank(b, loc, 1, 4);
            Value computed =
                b.create<BankMatrixOp>(loc, accB.getType(), patch, fQuant, accB,
                                       createI64Const(b, loc, (int64_t)cfg));
            releaseBank(b, loc, patch);
            releaseBank(b, loc, fQuant);
            mvoutBank(b, loc, tmpI, computed, cBlocks);
            releaseBank(b, loc, computed);

            Value tF = allocBank(b, loc, 1, 4);
            Value tL = mvinBank(b, loc, tmpI, tF, cBlocks);
            Value fp =
                b.create<BankInt2FpOp>(loc, tL.getType(), tL, tL,
                                       createI64Const(b, loc, cBlocks), scaleD);
            mvoutBank(b, loc, tmpF, fp, cBlocks);
            releaseBank(b, loc, fp);
            addF32Pack(b, loc, accF, tmpF, cBlocks);
            b.setInsertionPointAfter(cinL);

            Value tileV = b.create<arith::ConstantIndexOp>(loc, tile);
            Value oh0V = b.create<arith::ConstantIndexOp>(loc, oh0);
            Value ow0V = b.create<arith::ConstantIndexOp>(loc, ow0);
            Value oc0V = b.create<arith::ConstantIndexOp>(loc, oc0);
            auto ohL = b.create<scf::ForOp>(loc, zero, tileV, one);
            b.setInsertionPointToStart(ohL.getBody());
            auto owL = b.create<scf::ForOp>(loc, zero, tileV, one);
            b.setInsertionPointToStart(owL.getBody());
            Value win = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, ohL.getInductionVar(), tileV),
                owL.getInductionVar());
            auto ocL = b.create<scf::ForOp>(
                loc, zero, b.create<arith::ConstantIndexOp>(loc, ocTile), one);
            b.setInsertionPointToStart(ocL.getBody());
            Value ov = b.create<memref::LoadOp>(
                loc, accF, ValueRange{win, ocL.getInductionVar()});
            Value oGlobal =
                b.create<arith::AddIOp>(loc, oc0V, ocL.getInductionVar());
            Value oh =
                b.create<arith::AddIOp>(loc, oh0V, ohL.getInductionVar());
            Value ow =
                b.create<arith::AddIOp>(loc, ow0V, owL.getInductionVar());
            SmallVector<Value, 4> outIdx{nV, oh, ow, oGlobal};
            Value cur = b.create<memref::LoadOp>(loc, op.getOutput(), outIdx);
            b.create<memref::StoreOp>(loc,
                                      b.create<arith::AddFOp>(loc, cur, ov),
                                      op.getOutput(), outIdx);
            b.setInsertionPointAfter(ohL);
          }
        }
      }
    }

    b.create<memref::DeallocOp>(loc, inPack);
    b.create<memref::DeallocOp>(loc, fPack);
    b.create<memref::DeallocOp>(loc, tmpI);
    b.create<memref::DeallocOp>(loc, tmpF);
    b.create<memref::DeallocOp>(loc, accF);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  populateMatrixTileMatMulPatterns(patterns, bankWidthBytes, bankDepth,
                                   bankNum);
  patterns.add<TileTransposeLowering>(patterns.getContext(), bankWidthBytes);
  patterns.add<TileConv2dLowering>(patterns.getContext());
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
