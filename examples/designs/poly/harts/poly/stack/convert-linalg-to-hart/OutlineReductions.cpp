//===- OutlineReductions.cpp - Row reductions as GP domain operators
//--------===//
//
// Sums and maxima over FP32 rows become GP operators. Both dimensions
// may be dynamic; each row retains its initial value and ordered sums.
// A four-float output slot keeps DMA aligned, with only its first value used.
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

LogicalResult outlineReductions(ModuleOp source) {
  SmallVector<std::pair<linalg::GenericOp, bool>> reductions;
  source.walk([&](linalg::GenericOp op) {
    if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 1 ||
        op.getNumResults() != 1 || op.getNumLoops() < 2 ||
        op.getNumParallelLoops() + 1 != op.getNumLoops() ||
        op.getIteratorTypesArray().back() != utils::IteratorType::reduction)
      return;
    auto input = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
    auto output = cast<RankedTensorType>(op.getResult(0).getType());
    if (!input.getElementType().isF32() ||
        (!input.isDynamicDim(input.getRank() - 1) &&
         input.getShape().back() <= 0) ||
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
    Operation *combine = body.getTerminator()->getOperand(0).getDefiningOp();
    if (!isa_and_nonnull<arith::AddFOp, arith::MaximumFOp>(combine))
      return;
    Value accumulator = body.getArgument(1);
    Value term = combine->getOperand(0) == accumulator ? combine->getOperand(1)
                                                       : combine->getOperand(0);
    if (combine->getOperand(0) != accumulator &&
        combine->getOperand(1) != accumulator)
      return;
    auto square = term.getDefiningOp<arith::MulFOp>();
    bool squared = square && square.getLhs() == body.getArgument(0) &&
                   square.getRhs() == body.getArgument(0);
    if ((term != body.getArgument(0) && !squared) ||
        body.getOperations().size() != (squared ? 3 : 2))
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
    reductions.emplace_back(op, squared);
  });

  std::map<std::tuple<int64_t, bool, bool, uint64_t>, StringAttr> operators;
  for (auto [op, squared] : reductions) {
    OpBuilder b(source.getContext());
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
    int64_t width = inputType.getShape().back();
    int64_t rows = 1;
    for (int64_t axis = 0; axis < inputType.getRank() - 1; ++axis)
      rows = rows < 0 || inputType.isDynamicDim(axis)
                 ? ShapedType::kDynamic
                 : rows * inputType.getDimSize(axis);
    bool maximum = isa<arith::MaximumFOp>(
        op.getRegion().front().getTerminator()->getOperand(0).getDefiningOp());
    auto initial = cast<FloatAttr>(op.getDpsInits()[0]
                                       .getDefiningOp<linalg::FillOp>()
                                       .getDpsInputs()[0]
                                       .getDefiningOp<arith::ConstantOp>()
                                       .getValue());
    auto [entry, fresh] = operators.try_emplace(
        std::make_tuple(width, maximum, squared,
                        initial.getValue().bitcastToAPInt().getZExtValue()));
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    if (fresh) {
      auto function = createOperator(b, source, loc);
      entry->second = function.getSymNameAttr();
      Value input = operatorMemory(b, loc, function.getArgument(0),
                                   MemRefType::get({width}, b.getF32Type()));
      Value results = operatorMemory(b, loc, function.getArgument(1),
                                     MemRefType::get({4}, b.getF32Type()));
      Value initialValue = arith::ConstantOp::create(b, loc, initial);
      Value extent =
          width < 0 ? Value(arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                       function.getArgument(6)))
                    : index(width);
      Value bulk = b.createOrFold<arith::MulIOp>(
          loc, b.createOrFold<arith::DivUIOp>(loc, extent, index(32)),
          index(32));
      auto loop = scf::ForOp::create(b, loc, index(0), bulk, index(32),
                                     ValueRange{initialValue});
      b.setInsertionPointToStart(loop.getBody());
      Value values =
          vector::LoadOp::create(b, loc, VectorType::get({32}, b.getF32Type()),
                                 input, ValueRange{loop.getInductionVar()});
      if (squared)
        values = arith::MulFOp::create(b, loc, values, values);
      Value sum =
          vector::ReductionOp::create(b, loc,
                                      maximum ? vector::CombiningKind::MAXIMUMF
                                              : vector::CombiningKind::ADD,
                                      values, loop.getRegionIterArgs()[0]);
      scf::YieldOp::create(b, loc, sum);
      b.setInsertionPointAfter(loop);
      auto tail = scf::ForOp::create(b, loc, bulk, extent, index(1),
                                     ValueRange{loop.getResult(0)});
      b.setInsertionPointToStart(tail.getBody());
      Value value = memref::LoadOp::create(b, loc, input,
                                           ValueRange{tail.getInductionVar()});
      if (squared)
        value = arith::MulFOp::create(b, loc, value, value);
      sum = maximum ? Value(arith::MaximumFOp::create(
                          b, loc, tail.getRegionIterArgs()[0], value))
                    : Value(arith::AddFOp::create(
                          b, loc, tail.getRegionIterArgs()[0], value));
      scf::YieldOp::create(b, loc, sum);
      b.setInsertionPointAfter(tail);
      memref::StoreOp::create(b, loc, tail.getResult(0), results,
                              ValueRange{index(0)});
      Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
      for (int64_t padding = 1; padding < 4; ++padding)
        memref::StoreOp::create(b, loc, zero, results,
                                ValueRange{index(padding)});
      func::ReturnOp::create(b, loc);
    }

    b.setInsertionPoint(op);
    SmallVector<OpFoldResult> dimensions =
        tensor::getMixedSizes(b, loc, op.getDpsInputs()[0]);
    dimensions.pop_back();
    ReassociationIndices inputAxes;
    for (int64_t axis = 0; axis < inputType.getRank() - 1; ++axis)
      inputAxes.push_back(axis);
    Value grouped = op.getDpsInputs()[0];
    if (inputType.getRank() > 2)
      grouped = tensor::CollapseShapeOp::create(
          b, loc, RankedTensorType::get({rows, width}, b.getF32Type()), grouped,
          ArrayRef<ReassociationIndices>{inputAxes, {inputType.getRank() - 1}});
    Value rowCount = tensor::DimOp::create(b, loc, grouped, 0);
    auto map = AffineMap::getMultiDimIdentityMap(2, source.getContext());
    auto call = ::buddy::acc::CustomOp::create(
        b, loc, TypeRange{RankedTensorType::get({rows, 4}, b.getF32Type())},
        "row_operator", ValueRange{grouped});
    call->setAttr("operator", FlatSymbolRefAttr::get(entry->second));
    call->setAttr("maps", b.getAffineMapArrayAttr({map, map}));
    call->setAttr("input_offsets", b.getDenseI64ArrayAttr({0}));
    call->setAttr("input_widths", b.getDenseI64ArrayAttr({width}));
    if (width < 0)
      call->setAttr("width_input", b.getI64IntegerAttr(0));
    Value result = tensor::ExtractSliceOp::create(
        b, loc, RankedTensorType::get({rows, 1}, b.getF32Type()),
        call.getResult(0),
        ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{rows < 0 ? OpFoldResult(rowCount)
                                        : OpFoldResult(b.getIndexAttr(rows)),
                               b.getIndexAttr(1)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value linear = tensor::CollapseShapeOp::create(
        b, loc, RankedTensorType::get({rows}, b.getF32Type()), result,
        ArrayRef<ReassociationIndices>{{0, 1}});
    auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
    Value expanded = linear;
    if (outputType.getRank() > 1)
      expanded = tensor::ExpandShapeOp::create(
          b, loc, outputType, linear, ArrayRef<ReassociationIndices>{inputAxes},
          dimensions);
    op.getResult(0).replaceAllUsesWith(expanded);
    op.erase();
  }
  return success();
}
