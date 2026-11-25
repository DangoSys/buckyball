//===- MatmulConversion.cpp - The GP operator of an accumulator tile
//--------===//
//
// Converts the rows of an INT32 accumulator tile (argument 0) to FP32 and
// scales them, scattering each four-column group into its own output group
// (argument 2); argument 3 is the number of valid rows. The factors
// (argument 1) are one row of sixteen for all rows, or with `rowFactors` one
// row per row in the grouped layout of the output.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

func::FuncOp createOperator(OpBuilder &b, ModuleOp module, Location loc);
Value operatorMemory(OpBuilder &b, Location loc, Value address,
                     MemRefType type);

func::FuncOp createMatmulConversion(ModuleOp module, bool rowFactors) {
  OpBuilder b(module.getContext());
  Location loc = module.getLoc();
  func::FuncOp function = createOperator(b, module, loc);
  auto f32 = b.getF32Type();
  auto i64 = b.getI64Type();
  auto bankGroups = MemRefType::get({INT64_C(4) << 30}, f32);
  Value sumsMemory =
      operatorMemory(b, loc, function.getArgument(0),
                     MemRefType::get({INT64_C(1) << 30}, b.getI32Type()));
  Value factorMemory =
      operatorMemory(b, loc, function.getArgument(1), bankGroups);
  Value outputMemory =
      operatorMemory(b, loc, function.getArgument(2), bankGroups);
  auto index = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  auto values = VectorType::get({16}, f32);
  auto integers = VectorType::get({16}, b.getI32Type());
  auto indices = VectorType::get({16}, i64);
  auto maskType = VectorType::get({16}, b.getI1Type());
  SmallVector<APInt> offsets;
  for (int64_t column = 0; column < 16; ++column)
    offsets.emplace_back(64, ((column / 4) << 30) + column % 4);
  Value striped = arith::ConstantOp::create(
      b, loc, indices, DenseIntElementsAttr::get(indices, offsets));
  Value mask = arith::ConstantOp::create(
      b, loc, maskType, DenseElementsAttr::get(maskType, b.getBoolAttr(true)));
  Value zero = arith::ConstantOp::create(
      b, loc, values, DenseElementsAttr::get(values, b.getF32FloatAttr(0)));
  Value factors;
  if (!rowFactors)
    factors = vector::LoadOp::create(b, loc, values, factorMemory,
                                     ValueRange{index(0)});
  Value count = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                           function.getArgument(3));
  auto loop = scf::ForOp::create(b, loc, index(0), count, index(1));
  b.setInsertionPointToStart(loop.getBody());
  Value row = loop.getInductionVar();
  Value source = arith::MulIOp::create(b, loc, row, index(16));
  Value output = arith::MulIOp::create(b, loc, row, index(4));
  Value sums =
      vector::LoadOp::create(b, loc, integers, sumsMemory, ValueRange{source});
  Value converted = arith::SIToFPOp::create(b, loc, values, sums);
  if (rowFactors)
    factors = vector::GatherOp::create(b, loc, values, factorMemory,
                                       ValueRange{output}, striped, mask, zero);
  Value scaled = math::FmaOp::create(b, loc, converted, factors, zero);
  vector::ScatterOp::create(b, loc, Type(), outputMemory, ValueRange{output},
                            striped, mask, scaled);
  b.setInsertionPointAfter(loop);
  func::ReturnOp::create(b, loc);
  return function;
}
