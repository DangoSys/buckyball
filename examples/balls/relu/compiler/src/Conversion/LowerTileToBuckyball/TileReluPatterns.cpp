#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace {
class TileReluLowering : public OpRewritePattern<tile::TileReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileReluOp op,
                                PatternRewriter &b) const override {
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        inputType != outputType || inputType.getRank() != 2 ||
        !inputType.getElementType().isInteger(32))
      return op.emitError("requires matching static memref<MxNxi32>");

    int64_t rows = inputType.getShape()[0];
    int64_t columns = inputType.getShape()[1];
    if (rows <= 0 || columns <= 0)
      return op.emitError("requires positive dimensions");

    int64_t paddedRows = llvm::alignTo(rows, 16);
    int64_t paddedColumns = llvm::alignTo(columns, 16);
    Location loc = op.getLoc();
    if (paddedRows == rows && paddedColumns == columns) {
      b.create<ReluMatrixOp>(loc, op.getInput(), op.getOutput());
      b.eraseOp(op);
      return success();
    }

    auto paddedType =
        MemRefType::get({paddedRows, paddedColumns}, b.getI32Type());
    Value paddedInput = b.create<memref::AllocOp>(loc, paddedType);
    Value paddedOutput = b.create<memref::AllocOp>(loc, paddedType);
    Value zeroValue = b.create<arith::ConstantIntOp>(loc, 0, 32);
    b.create<linalg::FillOp>(loc, zeroValue, paddedInput);

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value rowLimit = b.create<arith::ConstantIndexOp>(loc, rows);
    Value columnLimit = b.create<arith::ConstantIndexOp>(loc, columns);
    auto copyInputRows = b.create<scf::ForOp>(loc, zero, rowLimit, one);
    b.setInsertionPointToStart(copyInputRows.getBody());
    Value row = copyInputRows.getInductionVar();
    auto copyInputColumns = b.create<scf::ForOp>(loc, zero, columnLimit, one);
    b.setInsertionPointToStart(copyInputColumns.getBody());
    Value column = copyInputColumns.getInductionVar();
    Value value =
        b.create<memref::LoadOp>(loc, op.getInput(), ValueRange{row, column});
    b.create<memref::StoreOp>(loc, value, paddedInput, ValueRange{row, column});
    b.setInsertionPointAfter(copyInputRows);

    b.create<ReluMatrixOp>(loc, paddedInput, paddedOutput);

    auto copyOutputRows = b.create<scf::ForOp>(loc, zero, rowLimit, one);
    b.setInsertionPointToStart(copyOutputRows.getBody());
    row = copyOutputRows.getInductionVar();
    auto copyOutputColumns = b.create<scf::ForOp>(loc, zero, columnLimit, one);
    b.setInsertionPointToStart(copyOutputColumns.getBody());
    column = copyOutputColumns.getInductionVar();
    value =
        b.create<memref::LoadOp>(loc, paddedOutput, ValueRange{row, column});
    b.create<memref::StoreOp>(loc, value, op.getOutput(),
                              ValueRange{row, column});
    b.setInsertionPointAfter(copyOutputRows);

    b.create<memref::DeallocOp>(loc, paddedInput);
    b.create<memref::DeallocOp>(loc, paddedOutput);
    b.eraseOp(op);
    return success();
  }
};
} // namespace

namespace mlir::buddy {
void populateReluBallTileLoweringPatterns(RewritePatternSet &patterns, int64_t,
                                          int64_t, int64_t) {
  patterns.add<TileReluLowering>(patterns.getContext());
}
} // namespace mlir::buddy
