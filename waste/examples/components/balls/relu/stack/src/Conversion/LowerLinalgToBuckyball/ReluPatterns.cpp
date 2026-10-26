//===- ReluPatterns.cpp - signed INT32 ReLU -> buckyball.relu_matrix -----===//
//
// Claims linalg.generic max(input, 0) over static memref<MxNxi32>.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
// The zero operand must be a constant zero splat global.
bool isZeroGlobal(Value value) {
  auto global = value.getDefiningOp<memref::GetGlobalOp>();
  if (!global)
    return false;
  auto constant = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      global, global.getNameAttr());
  auto values =
      constant
          ? dyn_cast_or_null<DenseElementsAttr>(constant.getConstantInitValue())
          : nullptr;
  return values && values.isSplat() && values.getSplatValue<APInt>().isZero();
}

class ReluLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &b) const override {
    Block &body = op.getRegion().front();
    if (body.getOperations().size() != 2 || op.getInputs().size() != 2 ||
        op.getOutputs().size() != 1)
      return failure();
    auto maximum = dyn_cast<arith::MaxSIOp>(body.front());
    ValueRange args = body.getArguments();
    if (!maximum || body.back().getOperand(0) != maximum.getResult() ||
        !((maximum.getLhs() == args[0] && maximum.getRhs() == args[1]) ||
          (maximum.getLhs() == args[1] && maximum.getRhs() == args[0])))
      return failure();
    Value input = op.getInputs()[0];
    Value output = op.getOutputs()[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 2 ||
        !inputType.getElementType().isInteger(32) ||
        op.getInputs()[1].getType() != inputType ||
        output.getType() != inputType || !isZeroGlobal(op.getInputs()[1]) ||
        !llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }))
      return failure();

    int64_t rows = inputType.getShape()[0];
    int64_t columns = inputType.getShape()[1];
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
void populateReluBallLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<ReluLowering>(patterns.getContext());
}
} // namespace mlir::buddy
