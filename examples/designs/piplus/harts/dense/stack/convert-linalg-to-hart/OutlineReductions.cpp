//===- OutlineReductions.cpp - Row reductions as GP domain operators
//--------===//
//
// Sums and maxima over the last dimension of FP32 tensors (a row within a
// group of the bank) become row_operators. With row count and width multiples
// of four, on [M / 4, 4 * W], its four rows its four inputs (one group
// each): each operator reduces four rows. Otherwise on [M, W], each operator
// reducing one row into the four lanes of a 16-byte row, [M, 4], of which
// the first is the result. A row is reduced in order from the initial value
// across chunks.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <map>
#include <tuple>

using namespace mlir;

func::FuncOp createOperator(OpBuilder &b, ModuleOp module, Location loc);
Value operatorMemory(OpBuilder &b, Location loc, Value address,
                     MemRefType type);

LogicalResult outlineReductions(ModuleOp source, int64_t group) {
  SmallVector<linalg::ReduceOp> named;
  source.walk([&](linalg::ReduceOp op) { named.push_back(op); });
  IRRewriter rewriter(source.getContext());
  for (auto op : named) {
    rewriter.setInsertionPoint(op);
    if (failed(linalg::generalizeNamedOp(rewriter, op)))
      return op.emitError("cannot generalize reduction");
  }
  SmallVector<linalg::GenericOp> reductions;
  source.walk([&](linalg::GenericOp op) {
    if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 1 ||
        op.getNumResults() != 1 || op.getNumLoops() < 2 ||
        op.getNumParallelLoops() + 1 != op.getNumLoops() ||
        op.getIteratorTypesArray().back() != utils::IteratorType::reduction)
      return;
    auto input = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
    auto output = cast<RankedTensorType>(op.getResult(0).getType());
    if (!input.hasStaticShape() || !input.getElementType().isF32() ||
        input.getShape().back() < 16 || input.getShape().back() > group ||
        output.getShape() != input.getShape().drop_back())
      return;
    SmallVector<unsigned> axes;
    for (unsigned axis = 0; axis < input.getRank() - 1; ++axis)
      axes.push_back(axis);
    auto maps = op.getIndexingMapsArray();
    if (!maps[0].isIdentity() ||
        maps[1] != AffineMap::getMultiDimIdentityMap(input.getRank(),
                                                     source.getContext())
                       .getSubMap(axes))
      return;
    Block &body = op.getRegion().front();
    Operation *combine = &body.front();
    if (body.getOperations().size() != 2 ||
        !isa<arith::AddFOp, arith::MaximumFOp>(combine) ||
        body.getTerminator()->getOperand(0) != combine->getResult(0) ||
        !((combine->getOperand(0) == body.getArgument(0) &&
           combine->getOperand(1) == body.getArgument(1)) ||
          (combine->getOperand(0) == body.getArgument(1) &&
           combine->getOperand(1) == body.getArgument(0))))
      return;
    auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fill)
      return;
    auto constant = fill.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
    if (!constant ||
        (isa<arith::AddFOp>(combine) && !cast<FloatAttr>(constant.getValue())
                                             .getValue()
                                             .bitcastToAPInt()
                                             .isZero()))
      return;
    reductions.push_back(op);
  });

  std::map<std::tuple<int64_t, bool, uint64_t, bool>, StringAttr> operators;
  for (auto op : reductions) {
    OpBuilder b(source.getContext());
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
    int64_t width = inputType.getShape().back();
    int64_t rows = inputType.getNumElements() / width;
    bool maximum = isa<arith::MaximumFOp>(op.getRegion().front().front());
    // Four rows an operator, or one.
    bool grouped = width % 4 == 0 && rows % 4 == 0;
    int64_t together = grouped ? 4 : 1;
    auto initial = cast<FloatAttr>(op.getDpsInits()[0]
                                       .getDefiningOp<linalg::FillOp>()
                                       .getDpsInputs()[0]
                                       .getDefiningOp<arith::ConstantOp>()
                                       .getValue());
    auto [entry, fresh] = operators.try_emplace(std::make_tuple(
        width, maximum, initial.getValue().bitcastToAPInt().getZExtValue(),
        grouped));
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    if (fresh) {
      auto function = createOperator(b, source, loc);
      entry->second = function.getSymNameAttr();
      Value results = operatorMemory(b, loc, function.getArgument(together),
                                     MemRefType::get({4}, b.getF32Type()));
      Value initialValue = arith::ConstantOp::create(b, loc, initial);
      for (int64_t row = 0; row < together; ++row) {
        Value input = operatorMemory(b, loc, function.getArgument(row),
                                     MemRefType::get({width}, b.getF32Type()));
        auto loop = scf::ForOp::create(b, loc, index(0), index(width / 32 * 32),
                                       index(32), ValueRange{initialValue});
        b.setInsertionPointToStart(loop.getBody());
        Value values = vector::LoadOp::create(
            b, loc, VectorType::get({32}, b.getF32Type()), input,
            ValueRange{loop.getInductionVar()});
        // Carry the previous chunk's result and retain the original initial
        // value. MAXIMUMF includes NaN propagation and signed-zero ordering.
        Value sum = vector::ReductionOp::create(
            b, loc,
            maximum ? vector::CombiningKind::MAXIMUMF
                    : vector::CombiningKind::ADD,
            values, loop.getRegionIterArgs()[0]);
        scf::YieldOp::create(b, loc, sum);
        b.setInsertionPointAfter(loop);
        sum = loop.getResult(0);
        if (width % 32) {
          auto tail =
              scf::ForOp::create(b, loc, index(width / 32 * 32), index(width),
                                 index(1), ValueRange{sum});
          b.setInsertionPointToStart(tail.getBody());
          Value value = memref::LoadOp::create(
              b, loc, input, ValueRange{tail.getInductionVar()});
          Value carried = tail.getRegionIterArgs()[0];
          scf::YieldOp::create(
              b, loc,
              maximum ? Value(arith::MaximumFOp::create(b, loc, carried, value))
                      : Value(arith::AddFOp::create(b, loc, carried, value)));
          b.setInsertionPointAfter(tail);
          sum = tail.getResult(0);
        }
        for (int64_t lane = grouped ? row : 0; lane < (grouped ? row + 1 : 4);
             ++lane)
          memref::StoreOp::create(b, loc, sum, results,
                                  ValueRange{index(lane)});
      }
      func::ReturnOp::create(b, loc);
    }

    b.setInsertionPoint(op);
    ReassociationIndices inputAxes;
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis)
      inputAxes.push_back(axis);
    Value flattened = tensor::CollapseShapeOp::create(
        b, loc, RankedTensorType::get({rows * width}, b.getF32Type()),
        op.getDpsInputs()[0], ArrayRef<ReassociationIndices>{inputAxes});
    Value rowsOf = tensor::ExpandShapeOp::create(
        b, loc,
        RankedTensorType::get({rows / together, together * width},
                              b.getF32Type()),
        flattened, ArrayRef<ReassociationIndices>{{0, 1}});
    auto map = AffineMap::getMultiDimIdentityMap(2, source.getContext());
    SmallVector<Value> inputs(together, rowsOf);
    SmallVector<AffineMap> maps(together + 1, map);
    SmallVector<int64_t> offsets, widths(together, width);
    for (int64_t row = 0; row < together; ++row)
      offsets.push_back(row * width);
    auto call = ::buddy::acc::CustomOp::create(
        b, loc,
        TypeRange{RankedTensorType::get({rows / together, 4}, b.getF32Type())},
        "row_operator", inputs);
    call->setAttr("operator", FlatSymbolRefAttr::get(entry->second));
    call->setAttr("maps", b.getAffineMapArrayAttr(maps));
    call->setAttr("input_offsets", b.getDenseI64ArrayAttr(offsets));
    call->setAttr("input_widths", b.getDenseI64ArrayAttr(widths));
    call->setAttr("chunk", b.getI64IntegerAttr(4));
    call->setAttr("rows", b.getI64IntegerAttr(1));
    Value result = call.getResult(0);
    if (!grouped)
      result = tensor::ExtractSliceOp::create(
          b, loc, RankedTensorType::get({rows, 1}, b.getF32Type()), result,
          ValueRange{}, ValueRange{}, ValueRange{},
          b.getDenseI64ArrayAttr({0, 0}), b.getDenseI64ArrayAttr({rows, 1}),
          b.getDenseI64ArrayAttr({1, 1}));
    Value linear = tensor::CollapseShapeOp::create(
        b, loc, RankedTensorType::get({rows}, b.getF32Type()), result,
        ArrayRef<ReassociationIndices>{{0, 1}});
    auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
    inputAxes.pop_back();
    Value expanded = linear;
    if (outputType.getRank() > 1)
      expanded = tensor::ExpandShapeOp::create(
          b, loc, outputType, linear,
          ArrayRef<ReassociationIndices>{inputAxes});
    op.getResult(0).replaceAllUsesWith(expanded);
    op.erase();
  }
  return success();
}
