#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace mlir::buddy {
void populateTransposeBallTileLoweringPatterns(RewritePatternSet &patterns,
                                               int64_t bankWidthBytes,
                                               int64_t bankDepth,
                                               int64_t bankNum);
} // namespace mlir::buddy

namespace {
class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileTransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto outputType = dyn_cast<MemRefType>(op.getBMemArray().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || inputType.getRank() != 2 ||
        outputType.getRank() != 2)
      return op.emitError("requires static rank-2 input and output memrefs");
    if (outputType.getShape()[0] != inputType.getShape()[1] ||
        outputType.getShape()[1] != inputType.getShape()[0] ||
        inputType.getElementType() != outputType.getElementType())
      return op.emitError("requires a matching transposed output memref");
    rewriter.create<MemTransposeOp>(op.getLoc(), op.getAMemArray(),
                                    op.getBMemArray());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::buddy::populateTransposeBallTileLoweringPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  (void)bankWidthBytes;
  (void)bankDepth;
  (void)bankNum;
  patterns.add<TileTransposeLowering>(patterns.getContext());
}
