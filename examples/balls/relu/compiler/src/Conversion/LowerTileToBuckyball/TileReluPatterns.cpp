#include "Conversion/LowerTileToBuckyball/LowerTileToBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::tile;
using mlir::buddy::ceilDiv;
using mlir::buddy::kBankLane;

namespace {

class TileReluLowering : public OpRewritePattern<TileReluOp> {
public:
  using OpRewritePattern<TileReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileReluOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || inputType.getRank() != 2 ||
        outputType.getRank() != 2 ||
        !inputType.getElementType().isInteger(32) || outputType != inputType)
      return op.emitError("requires matching static memref<MxNxi32>");

    int64_t rows = inputType.getShape()[0];
    int64_t columns = inputType.getShape()[1];
    if (rows <= 0 || columns <= 0)
      return op.emitError("dimensions must be positive");

    int64_t paddedRows = ceilDiv(rows, (int64_t)kBankLane) * kBankLane;
    int64_t paddedColumns = ceilDiv(columns, (int64_t)kBankLane) * kBankLane;
    Value input = op.getInput();
    Value output = op.getOutput();

    if (paddedRows != rows || paddedColumns != columns) {
      Location loc = op.getLoc();
      auto paddedType = MemRefType::get({paddedRows, paddedColumns},
                                        inputType.getElementType());
      Value paddedInput = rewriter.create<memref::AllocOp>(loc, paddedType);
      Value paddedOutput = rewriter.create<memref::AllocOp>(loc, paddedType);
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, inputType.getElementType(),
          rewriter.getZeroAttr(inputType.getElementType()));
      rewriter.create<linalg::FillOp>(loc, zero, paddedInput);
      Value inputView = rewriter.create<memref::SubViewOp>(
          loc, paddedInput,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(rows),
                                    rewriter.getIndexAttr(columns)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, input, inputView);
      input = paddedInput;
      output = paddedOutput;

      rewriter.create<ReluMatrixOp>(loc, input, output);
      Value outputView = rewriter.create<memref::SubViewOp>(
          loc, output,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(rows),
                                    rewriter.getIndexAttr(columns)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)});
      rewriter.create<memref::CopyOp>(loc, outputView, op.getOutput());
      rewriter.create<memref::DeallocOp>(loc, paddedInput);
      rewriter.create<memref::DeallocOp>(loc, paddedOutput);
    } else {
      rewriter.create<ReluMatrixOp>(op.getLoc(), input, output);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populateReluBallTileLoweringPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  (void)bankWidthBytes;
  (void)bankDepth;
  (void)bankNum;
  patterns.add<TileReluLowering>(patterns.getContext());
}
