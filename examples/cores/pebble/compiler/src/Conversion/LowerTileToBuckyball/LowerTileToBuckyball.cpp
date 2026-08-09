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

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;
using mlir::buddy::kDefaultBankWidthBytes;
using mlir::buddy::populateMatrixTileMatMulPatterns;

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
// tile->buckyball: pack + buckyball.im2col_matmul (no quant). pad=0 stride=1.
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

            b.create<Im2colMatmulOp>(
                loc, inPack, fPack, tmpF, b.getI64IntegerAttr(inSize),
                b.getI64IntegerAttr(KH), b.getI64IntegerAttr(ocTile),
                b.getI64IntegerAttr(1), b.getI64IntegerAttr(0));
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
  patterns.add<TileTransposeLowering>(patterns.getContext());
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
