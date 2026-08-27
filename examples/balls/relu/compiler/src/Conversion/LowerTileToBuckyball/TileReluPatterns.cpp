#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
static bool getReluOperands(linalg::GenericOp generic, Value &input,
                            Value &output) {
  if (generic.getInputs().size() != 2 || generic.getOutputs().size() != 1)
    return false;
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  auto maximum = dyn_cast<arith::MaxSIOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!maximum || !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != maximum.getResult())
    return false;
  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 3 || !maps[0].isIdentity() || !maps[1].isIdentity() ||
      !maps[2].isIdentity())
    return false;
  for (utils::IteratorType iterator : generic.getIteratorTypesArray())
    if (iterator != utils::IteratorType::parallel)
      return false;
  ValueRange args = body.getArguments();
  if (args.size() != 3 ||
      !((maximum.getLhs() == args[0] && maximum.getRhs() == args[1]) ||
        (maximum.getLhs() == args[1] && maximum.getRhs() == args[0])))
    return false;

  Value zero = generic.getInputs()[1];
  auto global = zero.getDefiningOp<memref::GetGlobalOp>();
  if (!global)
    return false;
  auto constant = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      global, global.getNameAttr());
  auto values =
      constant
          ? dyn_cast_or_null<DenseElementsAttr>(constant.getConstantInitValue())
          : nullptr;
  if (!values || !values.isSplat() || !values.getSplatValue<APInt>().isZero())
    return false;

  input = generic.getInputs()[0];
  output = generic.getOutputs()[0];
  return true;
}

class TileReluLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &b) const override {
    Value input;
    Value output;
    if (!getReluOperands(op, input, output))
      return failure();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
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
      b.create<ReluMatrixOp>(loc, input, output);
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
    Value value = b.create<memref::LoadOp>(loc, input, ValueRange{row, column});
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
    b.create<memref::StoreOp>(loc, value, output, ValueRange{row, column});
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
