//===- Im2colMatmulToBankSSAPatterns.cpp - f32 im2col_matmul -> Bank* -----===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Utils/BankUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kLane = 16;
constexpr int64_t kBankDepth = 1024;
constexpr int64_t kMmioBytes = 5 * 1024;

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static LogicalResult rowStrideDivLane(MemRefType ty, int64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % kLane != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = strides[0] / kLane;
  return success();
}

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
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");
    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    if (stride < 1 || padding < 0 || startRow < 0 || startCol < 0)
      return op.emitError("stride/padding/start out of range");
    if (startRow > padding || startCol > padding)
      return op.emitError("startRow/Col must be <= padding");
    if (ksize < 1 || n < kLane || n > 4096 || n % kLane != 0)
      return op.emitError("ksize/n out of range");
    int64_t requiredDwBytes = perChannel ? n * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");

    int64_t kElems = ksize * ksize;
    int64_t padded = inSize + 2 * padding;
    if (padded < ksize + startRow || padded < ksize + startCol)
      return op.emitError("kernel+start larger than padded input");
    if ((padded - ksize - startRow) % stride != 0 ||
        (padded - ksize - startCol) % stride != 0)
      return op.emitError("inSize/pad/start/stride yield non-integer tile");
    int64_t tileH = (padded - ksize - startRow) / stride + 1;
    int64_t tileW = (padded - ksize - startCol) / stride + 1;
    if (tileH != tileW || tileH < 1)
      return op.emitError(
          "im2col_matmul requires square non-empty output tile");
    int64_t tile = tileH;
    int64_t wins = tile * tile;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t kTiles = cdiv(kElems, kLane);
    int64_t bRows = kTiles * kLane;
    int64_t aRows = cdiv(wins, kLane) * kTiles * kLane;
    if (aRows > kBankDepth)
      return op.emitError("im2col A layout exceeds bank depth");
    if (wins > kBankDepth)
      return op.emitError("im2col C rows exceed bank depth");

    if (inTy.getShape()[0] != inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != bRows || fTy.getShape()[1] != n ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != n)
      return op.emitError("im2col_matmul packed shape mismatch");

    int64_t strideF = 0, strideC = 0;
    if (failed(rowStrideDivLane(fTy, strideF)) ||
        failed(rowStrideDivLane(oTy, strideC)))
      return op.emitError(
          "filter/output need static strided<[row,1]> row%16==0");

    Value daAddr = createI64Const(b, loc, 0);

    Value inFB = allocBank(b, loc, 1, 4);
    Value inIB = allocBank(b, loc, 1, 1);
    Value loaded = mvinBank(b, loc, op.getInput(), inFB, inRows);
    Value quant =
        b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                               createI64Const(b, loc, inRows), daAddr);
    releaseBank(b, loc, loaded);

    Value patches = allocBank(b, loc, 1, 1);
    Value patch = b.create<BankIm2colOp>(
        loc, patches.getType(), quant, patches, createI64Const(b, loc, inSize),
        createI64Const(b, loc, ksize), createI64Const(b, loc, stride),
        createI64Const(b, loc, padding), b.getI64IntegerAttr(startRow),
        b.getI64IntegerAttr(startCol));
    releaseBank(b, loc, quant);

    Value fIB = allocBank(b, loc, 1, 1);
    Value accB = allocBank(b, loc, 1, 4);
    Value outB = allocBank(b, loc, 1, 4);
    uint64_t cfg = matrixRs2((uint64_t)wins, (uint64_t)kLane, (uint64_t)kElems);

    for (int64_t n0 = 0; n0 < n; n0 += kLane) {
      Value bTile = b.create<memref::SubViewOp>(
          loc, op.getFilter(),
          SmallVector<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(n0)},
          SmallVector<OpFoldResult>{b.getIndexAttr(bRows),
                                    b.getIndexAttr(kLane)},
          SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
      Value fLoaded = mvinBank(b, loc, bTile, fIB, bRows, strideF);
      Value gemm =
          createBankSMatMul(b, loc, accB.getType(), patch, fLoaded, accB,
                            createI64Const(b, loc, (int64_t)cfg), wins);
      Value dwAddr =
          createI64Const(b, loc, dwAddrBase + (perChannel ? n0 * 4 : 0));
      Value fp =
          perChannel
              ? b.create<BankInt2FpChannelOp>(loc, outB.getType(), gemm, outB,
                                              createI64Const(b, loc, wins),
                                              daAddr, dwAddr)
                    .getResult()
              : b.create<BankInt2FpTensorOp>(loc, outB.getType(), gemm, outB,
                                             createI64Const(b, loc, wins),
                                             daAddr, dwAddr)
                    .getResult();
      Value cTile = b.create<memref::SubViewOp>(
          loc, op.getOutput(),
          SmallVector<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(n0)},
          SmallVector<OpFoldResult>{b.getIndexAttr(wins),
                                    b.getIndexAttr(kLane)},
          SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
      mvoutBank(b, loc, cTile, fp, wins, strideC);
    }

    releaseBank(b, loc, patch);
    releaseBank(b, loc, fIB);
    releaseBank(b, loc, accB);
    releaseBank(b, loc, outB);
    b.create<FenceOp>(loc);

    b.eraseOp(op);
    return success();
  }
};

class Im2colFatMatmulToBankSSAPattern
    : public OpRewritePattern<Im2colFatMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colFatMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");
    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t nCin = op.getNCin();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    if (stride < 1 || padding < 0 || startRow < 0 || startCol < 0)
      return op.emitError("stride/padding/start out of range");
    if (startRow > padding || startCol > padding)
      return op.emitError("startRow/Col must be <= padding");
    if (ksize < 1 || n < kLane || n > 4096 || n % kLane != 0)
      return op.emitError("ksize/n out of range");
    int64_t requiredDwBytes = perChannel ? n * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");
    if (nCin < 1 || nCin > 256)
      return op.emitError("nCin out of range");
    int64_t kElems = ksize * ksize;
    int64_t padded = inSize + 2 * padding;
    if (padded < ksize + startRow || padded < ksize + startCol)
      return op.emitError("kernel+start larger than padded input");
    if ((padded - ksize - startRow) % stride != 0 ||
        (padded - ksize - startCol) % stride != 0)
      return op.emitError("inSize/pad/start/stride yield non-integer tile");
    int64_t tileH = (padded - ksize - startRow) / stride + 1;
    int64_t tileW = (padded - ksize - startCol) / stride + 1;
    if (tileH != tileW || tileH < 1)
      return op.emitError("im2col_fat_matmul requires square output tile");
    int64_t wins = tileH * tileW;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t kTiles = cdiv(kElems, kLane);
    int64_t bRows = kTiles * kLane;
    int64_t aRowsSingle = cdiv(wins, kLane) * kTiles * kLane;
    if (aRowsSingle > kBankDepth)
      return op.emitError("im2col A layout exceeds bank depth");
    if (wins > kBankDepth)
      return op.emitError("im2col C rows exceed bank depth");

    if (inTy.getShape()[0] != nCin * inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != nCin * bRows || fTy.getShape()[1] != n ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != n)
      return op.emitError("im2col_fat_matmul packed shape mismatch");

    int64_t strideF = 0;
    if (failed(rowStrideDivLane(fTy, strideF)))
      return op.emitError("filter needs static strided<[row,1]> row%16==0");

    Value daAddr = createI64Const(b, loc, 0);

    Value f0 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(0.0f));
    Value partial = b.create<memref::AllocOp>(
        loc, MemRefType::get({wins, n}, b.getF32Type()));

    Value inFB = allocBank(b, loc, 1, 4);
    Value inIB = allocBank(b, loc, 1, 1);
    Value patches = allocBank(b, loc, 1, 1);
    Value fIB = allocBank(b, loc, 1, 1);
    Value accB = allocBank(b, loc, 1, 4);
    Value outB = allocBank(b, loc, 1, 4);

    b.create<linalg::FillOp>(loc, f0, op.getOutput());
    uint64_t cfg = matrixRs2((uint64_t)wins, (uint64_t)kLane, (uint64_t)kElems);
    for (int64_t lc = 0; lc < nCin; ++lc) {
      Value plane = b.create<memref::SubViewOp>(
          loc, op.getInput(),
          ArrayRef<OpFoldResult>{b.getIndexAttr(lc * inRows),
                                 b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(inRows), b.getIndexAttr(kLane)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

      Value loaded = mvinBank(b, loc, plane, inFB, inRows);
      Value quant =
          b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                                 createI64Const(b, loc, inRows), daAddr);
      Value patch = b.create<BankIm2colOp>(
          loc, patches.getType(), quant, patches,
          createI64Const(b, loc, inSize), createI64Const(b, loc, ksize),
          createI64Const(b, loc, stride), createI64Const(b, loc, padding),
          b.getI64IntegerAttr(startRow), b.getI64IntegerAttr(startCol));

      for (int64_t n0 = 0; n0 < n; n0 += kLane) {
        Value bTile = b.create<memref::SubViewOp>(
            loc, op.getFilter(),
            SmallVector<OpFoldResult>{b.getIndexAttr(lc * bRows),
                                      b.getIndexAttr(n0)},
            SmallVector<OpFoldResult>{b.getIndexAttr(bRows),
                                      b.getIndexAttr(kLane)},
            SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        Value fLoaded = mvinBank(b, loc, bTile, fIB, bRows, strideF);
        Value gemm =
            createBankSMatMul(b, loc, accB.getType(), patch, fLoaded, accB,
                              createI64Const(b, loc, (int64_t)cfg), wins);
        Value dwAddr =
            createI64Const(b, loc, dwAddrBase + (perChannel ? n0 * 4 : 0));
        Value fp =
            perChannel
                ? b.create<BankInt2FpChannelOp>(loc, outB.getType(), gemm, outB,
                                                createI64Const(b, loc, wins),
                                                daAddr, dwAddr)
                      .getResult()
                : b.create<BankInt2FpTensorOp>(loc, outB.getType(), gemm, outB,
                                               createI64Const(b, loc, wins),
                                               daAddr, dwAddr)
                      .getResult();
        Value cTile = b.create<memref::SubViewOp>(
            loc, partial,
            SmallVector<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(n0)},
            SmallVector<OpFoldResult>{b.getIndexAttr(wins),
                                      b.getIndexAttr(kLane)},
            SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        mvoutBank(b, loc, cTile, fp, wins);
      }

      b.create<FenceOp>(loc);
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      Value winsV = b.create<arith::ConstantIndexOp>(loc, wins);
      Value nV = b.create<arith::ConstantIndexOp>(loc, n);
      auto rowLoop = b.create<scf::ForOp>(loc, zero, winsV, one);
      b.setInsertionPointToStart(rowLoop.getBody());
      auto colLoop = b.create<scf::ForOp>(loc, zero, nV, one);
      b.setInsertionPointToStart(colLoop.getBody());
      Value row = rowLoop.getInductionVar();
      Value col = colLoop.getInductionVar();
      Value accumulated =
          b.create<memref::LoadOp>(loc, op.getOutput(), ValueRange{row, col});
      Value addend =
          b.create<memref::LoadOp>(loc, partial, ValueRange{row, col});
      b.create<memref::StoreOp>(
          loc, b.create<arith::AddFOp>(loc, accumulated, addend),
          op.getOutput(), ValueRange{row, col});
      b.setInsertionPointAfter(rowLoop);
    }

    releaseBank(b, loc, inFB);
    releaseBank(b, loc, inIB);
    releaseBank(b, loc, patches);
    releaseBank(b, loc, fIB);
    releaseBank(b, loc, accB);
    releaseBank(b, loc, outB);
    b.create<FenceOp>(loc);

    b.create<memref::DeallocOp>(loc, partial);
    b.eraseOp(op);
    return success();
  }
};

class Im2colDepthwiseMatmulToBankSSAPattern
    : public OpRewritePattern<Im2colDepthwiseMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colDepthwiseMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto fTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto oTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return b.notifyMatchFailure(op, "requires static memrefs");
    if (!inTy.getElementType().isF32() || !fTy.getElementType().isInteger(8) ||
        !oTy.getElementType().isF32())
      return b.notifyMatchFailure(
          op,
          "requires FP32 activations, offline INT8 weights, and FP32 output");
    int64_t dwAddrBase = op.getDwAddr();
    int64_t dwBytes = op.getDwBytes();
    bool perChannel = op.getPerChannel();
    if (dwAddrBase < 16 || dwAddrBase % 4)
      return op.emitError(
          "requires a 4-byte-aligned Dw MMIO byte address from RAX metadata");

    int64_t inSize = op.getInSize();
    int64_t ksize = op.getKsize();
    int64_t n = op.getN();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    if (stride < 1 || padding < 0 || startRow < 0 || startCol < 0)
      return op.emitError("stride/padding/start out of range");
    if (startRow > padding || startCol > padding)
      return op.emitError("startRow/Col must be <= padding");
    if (ksize < 1 || n < 1 || n > kLane)
      return op.emitError("ksize/n out of range");
    int64_t requiredDwBytes = perChannel ? n * 4 : 4;
    if (dwBytes < requiredDwBytes || dwAddrBase + requiredDwBytes > kMmioBytes)
      return op.emitError("Dw scale range exceeds Pebble MMIO space");

    int64_t kElems = ksize * ksize;
    int64_t padded = inSize + 2 * padding;
    if (padded < ksize + startRow || padded < ksize + startCol)
      return op.emitError("kernel+start larger than padded input");
    if ((padded - ksize - startRow) % stride != 0 ||
        (padded - ksize - startCol) % stride != 0)
      return op.emitError("inSize/pad/start/stride yield non-integer tile");
    int64_t tileH = (padded - ksize - startRow) / stride + 1;
    int64_t tileW = (padded - ksize - startCol) / stride + 1;
    if (tileH != tileW || tileH < 1)
      return op.emitError("requires square non-empty output tile");
    int64_t tile = tileH;
    int64_t wins = tile * tile;
    int64_t inRows = cdiv(inSize * inSize, kLane);
    int64_t bRows = cdiv(kElems, kLane) * kLane;
    if (wins > kBankDepth)
      return op.emitError("im2col_depthwise_matmul output exceeds bank depth");

    if (inTy.getShape()[0] != n * inRows || inTy.getShape()[1] != kLane ||
        fTy.getShape()[0] != n * bRows || fTy.getShape()[1] != kLane ||
        oTy.getShape()[0] != wins || oTy.getShape()[1] != kLane)
      return op.emitError("im2col_depthwise_matmul packed shape mismatch");

    uint64_t cfg = matrixRs2((uint64_t)wins, 1, (uint64_t)kElems);

    Value daAddr = createI64Const(b, loc, 0);

    Value f0 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value fOne = b.create<memref::AllocOp>(
        loc, MemRefType::get({bRows, kLane}, b.getI8Type()));
    Value tmpOut = b.create<memref::AllocOp>(
        loc, MemRefType::get({wins, kLane}, b.getF32Type()));

    Value inFB = allocBank(b, loc, 1, 4);
    Value inIB = allocBank(b, loc, 1, 1);
    Value patches = allocBank(b, loc, 1, 1);
    Value fIB = allocBank(b, loc, 1, 1);
    Value accB = allocBank(b, loc, 1, 4);
    Value outB = allocBank(b, loc, 1, 4);

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value winsV = b.create<arith::ConstantIndexOp>(loc, wins);

    for (int64_t lc = 0; lc < n; ++lc) {
      Value plane = b.create<memref::SubViewOp>(
          loc, op.getInput(),
          ArrayRef<OpFoldResult>{b.getIndexAttr(lc * inRows),
                                 b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(inRows), b.getIndexAttr(kLane)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

      b.create<linalg::FillOp>(loc, f0, fOne);
      for (int64_t k = 0; k < kElems; ++k) {
        Value rowV = b.create<arith::ConstantIndexOp>(loc, k);
        Value srcRow = b.create<arith::ConstantIndexOp>(loc, lc * bRows + k);
        Value colV = b.create<arith::ConstantIndexOp>(loc, lc);
        Value wt = b.create<memref::LoadOp>(loc, op.getFilter(),
                                            ValueRange{srcRow, colV});
        b.create<memref::StoreOp>(loc, wt, fOne, ValueRange{rowV, zero});
      }

      Value loaded = mvinBank(b, loc, plane, inFB, inRows);
      Value quant =
          b.create<BankFp2IntOp>(loc, inIB.getType(), loaded, inIB,
                                 createI64Const(b, loc, inRows), daAddr);
      Value patch = b.create<BankIm2colOp>(
          loc, patches.getType(), quant, patches,
          createI64Const(b, loc, inSize), createI64Const(b, loc, ksize),
          createI64Const(b, loc, stride), createI64Const(b, loc, padding),
          b.getI64IntegerAttr(startRow), b.getI64IntegerAttr(startCol));

      Value fLoaded = mvinBank(b, loc, fOne, fIB, bRows);
      Value computed =
          createBankSMatMul(b, loc, accB.getType(), patch, fLoaded, accB,
                            createI64Const(b, loc, (int64_t)cfg), wins);
      Value dwAddr =
          createI64Const(b, loc, dwAddrBase + (perChannel ? lc * 4 : 0));
      Value fp = perChannel ? b.create<BankInt2FpChannelOp>(
                                   loc, outB.getType(), computed, outB,
                                   createI64Const(b, loc, wins), daAddr, dwAddr)
                                  .getResult()
                            : b.create<BankInt2FpTensorOp>(
                                   loc, outB.getType(), computed, outB,
                                   createI64Const(b, loc, wins), daAddr, dwAddr)
                                  .getResult();
      mvoutBank(b, loc, tmpOut, fp, wins);
      b.create<FenceOp>(loc);

      Value lcV = b.create<arith::ConstantIndexOp>(loc, lc);
      auto rL = b.create<scf::ForOp>(loc, zero, winsV, one);
      b.setInsertionPointToStart(rL.getBody());
      Value r = rL.getInductionVar();
      Value v = b.create<memref::LoadOp>(loc, tmpOut, ValueRange{r, zero});
      b.create<memref::StoreOp>(loc, v, op.getOutput(), ValueRange{r, lcV});
      b.setInsertionPointAfter(rL);
    }

    releaseBank(b, loc, inFB);
    releaseBank(b, loc, inIB);
    releaseBank(b, loc, patches);
    releaseBank(b, loc, fIB);
    releaseBank(b, loc, accB);
    releaseBank(b, loc, outB);
    b.create<FenceOp>(loc);

    b.create<memref::DeallocOp>(loc, fOne);
    b.create<memref::DeallocOp>(loc, tmpOut);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populatePebbleIm2colMatmulToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<Im2colMatmulToBankSSAPattern>(patterns.getContext());
  patterns.add<Im2colFatMatmulToBankSSAPattern>(patterns.getContext());
  patterns.add<Im2colDepthwiseMatmulToBankSSAPattern>(patterns.getContext());
}
