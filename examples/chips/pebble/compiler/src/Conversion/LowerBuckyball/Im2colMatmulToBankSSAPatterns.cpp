//===- Im2colMatmulToBankSSAPatterns.cpp - f32 im2col_matmul -> Bank* -----===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Utils/BankUtils.h"
#include "Utils/QuantUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kLane = 16;

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

class Im2colMatmulToBankSSAPattern : public OpRewritePattern<Im2colMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isF32() ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(op, "pebble bank-ssa path is f32 only");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    if (stride != 1 || padding != 0)
      return op.emitError("only stride=1 padding=0 supported");
    if (ksize < 1 || n < 1 || n > kLane)
      return op.emitError("ksize/n out of range");

    int64_t kElems = ksize * ksize;
    int64_t tile = inSize - ksize + 1;
    if (tile < 1)
      return op.emitError("inSize/ksize yield empty output tile");
    int64_t wins = tile * tile;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t bRows = cdiv(kElems, kLane) * kLane;

    if (inTy.getShape()[0] != inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != bRows || fTy.getShape()[1] != kLane ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != kLane)
      return op.emitError("im2col_matmul packed shape mismatch");

    uint64_t cfg =
        packBits(wins, 0, 11) | packBits(n, 12, 23) | packBits(kElems, 24, 35);

    Value scaleAF =
        quantScale(b, loc, absMaxF32(b, loc, op.getInput(), inRows, kLane));
    Value scaleBF =
        quantScale(b, loc, absMaxF32(b, loc, op.getFilter(), bRows, kLane));
    Value scaleA = packF32BitsAsI64(b, loc, scaleAF);
    Value scaleB = packF32BitsAsI64(b, loc, scaleBF);
    Value scaleD =
        packF32BitsAsI64(b, loc, dequantScale(b, loc, scaleAF, scaleBF));

    Value tmpI = b.create<memref::AllocOp>(
        loc, MemRefType::get({wins, kLane}, b.getI32Type()));

    Value inFB = allocBank(b, loc, 1, 4);
    Value inIB = allocBank(b, loc, 1, 1);
    Value loaded = mvinBank(b, loc, op.getInput(), inFB, inRows);
    Value quant =
        b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                               createI64Const(b, loc, inRows), scaleA);
    releaseBank(b, loc, loaded);

    Value patches = allocBank(b, loc, 1, 1);
    Value patch = b.create<BankIm2colOp>(
        loc, patches.getType(), quant, patches, createI64Const(b, loc, inSize),
        createI64Const(b, loc, ksize), createI64Const(b, loc, stride),
        createI64Const(b, loc, padding));
    releaseBank(b, loc, quant);

    Value fFB = allocBank(b, loc, 1, 4);
    Value fIB = allocBank(b, loc, 1, 1);
    Value fLoaded = mvinBank(b, loc, op.getFilter(), fFB, bRows);
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
    mvoutBank(b, loc, tmpI, computed, wins);
    releaseBank(b, loc, computed);

    Value tF = allocBank(b, loc, 1, 4);
    Value tL = mvinBank(b, loc, tmpI, tF, wins);
    Value fp = b.create<BankInt2FpOp>(loc, tL.getType(), tL, tL,
                                      createI64Const(b, loc, wins), scaleD);
    mvoutBank(b, loc, op.getOutput(), fp, wins);
    releaseBank(b, loc, fp);
    b.create<memref::DeallocOp>(loc, tmpI);

    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populatePebbleIm2colMatmulToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<Im2colMatmulToBankSSAPattern>(patterns.getContext());
}
