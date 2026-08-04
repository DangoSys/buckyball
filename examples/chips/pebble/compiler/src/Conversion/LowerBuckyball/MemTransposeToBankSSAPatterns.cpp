//===- MemTransposeToBankSSAPatterns.cpp - mem_transpose -> Bank* ---------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Utils/BankUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kBankWidthBytes = 16;
constexpr int64_t kRowTile = 16;

static size_t elemsPerBankRow(Type elemType) {
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return 0;
  return kBankWidthBytes / (bitWidth / 8);
}

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

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

class MemTransposeToBankSSAPattern : public OpRewritePattern<MemTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MemTransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return op.emitError("requires static input and output memrefs");

    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();
    int64_t rows = inShape[0];
    int64_t cols = inShape[1];
    if (outShape[0] != cols || outShape[1] != rows)
      return op.emitError("output shape must transpose the input shape");

    Type elemTy = inputType.getElementType();
    int64_t elemBits = elemTy.getIntOrFloatBitWidth();
    if (elemBits != 8 && elemBits != 32)
      return op.emitError("only 8/32-bit transpose elements are supported");

    size_t elemsPerRow = elemsPerBankRow(elemTy);
    if (elemsPerRow == 0)
      return op.emitError("unsupported transpose element type");

    // src+dst share banks => at most 4 column-groups per side. Prefer the
    // smallest full-row tile that covers `cols` so small matrices (e.g. 16x16
    // i8) are not padded out to 64 and mvout-truncated.
    size_t maxColTile = elemsPerRow * 4;
    size_t colsAlign =
        ((size_t)cols + elemsPerRow - 1) / elemsPerRow * elemsPerRow;
    size_t colTile = colsAlign <= maxColTile ? colsAlign : maxColTile;
    if (colTile == 0)
      return op.emitError("tile width is smaller than one bank row");

    int64_t rowsPad = cdiv(rows, kRowTile) * kRowTile;
    int64_t colsPad = cdiv(cols, (int64_t)colTile) * (int64_t)colTile;
    bool pad = rowsPad != rows || colsPad != cols;
    int64_t dummyStride = 0;
    bool inContig = succeeded(rowStrideDiv16(inputType, dummyStride));
    bool outContig = succeeded(rowStrideDiv16(outputType, dummyStride));
    bool materialize = pad || !inContig || !outContig;

    Value inBuf = input;
    Value outBuf = output;
    if (materialize) {
      auto inPadTy = MemRefType::get({rowsPad, colsPad}, elemTy);
      auto outPadTy = MemRefType::get({colsPad, rowsPad}, elemTy);
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
    int64_t strideIn = colsPad / 16;
    int64_t strideOut = rowsPad / 16;
    for (int64_t r0 = 0; r0 < rowsPad; r0 += kRowTile) {
      for (int64_t c0 = 0; c0 < colsPad; c0 += (int64_t)colTile) {
        Value inTile = rewriter.create<memref::SubViewOp>(
            loc, inBuf,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(r0),
                                      rewriter.getIndexAttr(c0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(kRowTile),
                                      rewriter.getIndexAttr((int64_t)colTile)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, outBuf,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(c0),
                                      rewriter.getIndexAttr(r0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr((int64_t)colTile),
                                      rewriter.getIndexAttr(kRowTile)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        Value src = allocBank(rewriter, loc, 1, groups);
        Value dst = allocBank(rewriter, loc, 1, groups);
        Value loaded = mvinBank(rewriter, loc, inTile, src, kRowTile, strideIn);
        Value transposed = rewriter.create<BankTransposeOp>(
            loc, dst.getType(), loaded, dst,
            createI64Const(rewriter, loc, kRowTile),
            createI64Const(rewriter, loc, elemBits));
        // Output of transpose is [colTile, rowTile]; mvout depth = colTile.
        mvoutBank(rewriter, loc, outTile, transposed, (int64_t)colTile,
                  strideOut);
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
};

} // namespace

void mlir::buddy::populatePebbleMemTransposeToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MemTransposeToBankSSAPattern>(patterns.getContext());
}
