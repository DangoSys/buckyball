//===- OutlineOperators.cpp - GP domain operators of tensor regions
//-----------===//
//
// convert-linalg-to-hart first compiles the FP32 regions the GP domain of the
// matmul threads runs into operators of the module and replaces each region by
// a bbacc.custom "row_operator" that names its operator:
//
//   - a pointwise linalg.generic of static FP32 rows (the last dimension
//     at least sixteen, a multiple of four) whose inputs are rows or scalars
//     of a row (one per row or one for all), at most six inputs;
//   - a row sum (OutlineReductions.cpp).
//
// An operator is a private function of eight i64 arguments, the argument
// registers of the launch descriptor; it reaches its rows through @gp_space,
// the address space of the GP domain from address 0. The row operator reads
// its inputs from the groups of its read bank, input i at argument i (the
// group address, or the bits of a scalar) and writes its row to the address
// in the argument after them. convert-linalg-to-acc numbers the operators and
// adds their entry, `operator`, which runs operator a7.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

Value lowerErf(OpBuilder &b, Location loc, Value x);
LogicalResult outlineReductions(ModuleOp module);

// A new operator at the end of `module`, `b` at the start of its body.
func::FuncOp createOperator(OpBuilder &b, ModuleOp module, Location loc) {
  auto space = MemRefType::get({INT64_C(1) << 40}, b.getI8Type());
  b.setInsertionPointToEnd(module.getBody());
  if (!module.lookupSymbol<memref::GlobalOp>("gp_space"))
    memref::GlobalOp::create(b, loc, "gp_space", StringAttr(), space,
                             Attribute(), false, IntegerAttr());
  int64_t number = 0;
  while (module.lookupSymbol("operator_" + std::to_string(number)))
    ++number;
  SmallVector<Type> arguments(8, b.getI64Type());
  auto function =
      func::FuncOp::create(b, loc, "operator_" + std::to_string(number),
                           b.getFunctionType(arguments, {}));
  function.setPrivate();
  b.setInsertionPointToStart(function.addEntryBlock());
  return function;
}

// The memory of `type` at the GP address `address`.
Value operatorMemory(OpBuilder &b, Location loc, Value address,
                     MemRefType type) {
  auto space = MemRefType::get({INT64_C(1) << 40}, b.getI8Type());
  Value base = memref::GetGlobalOp::create(b, loc, space, "gp_space");
  Value shift = arith::IndexCastOp::create(b, loc, b.getIndexType(), address);
  return memref::ViewOp::create(b, loc, type, base, shift, ValueRange{});
}

// Part of convert-linalg-to-hart, before it places the graph.
LogicalResult outlineOperators(ModuleOp module) {
  // Fuse row arithmetic without expanding contractions through reshapes:
  // convert-linalg-to-hart must still see their matmul operations.
  RewritePatternSet patterns(module.getContext());
  linalg::populateElementwiseOpsFusionPatterns(
      patterns, [](OpOperand *operand) {
        Operation *producer = operand->get().getDefiningOp();
        return producer && producer->hasOneUse();
      });
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();
  SmallVector<linalg::GenericOp> regions;
  module.walk([&](linalg::GenericOp op) {
    if (!op.hasPureTensorSemantics() || op.getNumResults() != 1 ||
        op.getNumLoops() != op.getNumParallelLoops() ||
        op.getNumDpsInputs() > 6)
      return;
    auto output = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!output || !output.hasStaticShape() ||
        !output.getElementType().isF32() || output.getRank() == 0 ||
        output.getShape().back() < 16 || output.getShape().back() % 4 ||
        !op.getIndexingMapsArray().back().isIdentity() ||
        !op.getRegion().front().getArguments().back().use_empty())
      return;
    auto maps = op.getIndexingMapsArray();
    for (auto [input, map] : llvm::zip_equal(
             op.getDpsInputs(), ArrayRef<AffineMap>(maps).drop_back())) {
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
          type.getRank() == 0 || !map.isProjectedPermutation(true))
        return;
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      bool vector =
          isa<AffineDimExpr>(last) &&
          cast<AffineDimExpr>(last).getPosition() == op.getNumLoops() - 1;
      if (!vector && type.getShape().back() != 1)
        return;
      for (AffineExpr expression : map.getResults())
        if (auto constant = dyn_cast<AffineConstantExpr>(expression);
            constant && constant.getValue() != 0)
          return;
    }
    for (Operation &body : op.getRegion().front().without_terminator()) {
      StringRef name = body.getName().getStringRef();
      if (name != "arith.constant" && name != "arith.addf" &&
          name != "arith.subf" && name != "arith.mulf" &&
          name != "arith.divf" && name != "arith.negf" &&
          name != "arith.cmpf" && name != "arith.select" &&
          name != "arith.maximumf" && name != "arith.minimumf" &&
          name != "math.sqrt" && name != "math.rsqrt" && name != "math.absf" &&
          name != "math.erf" && name != "math.exp")
        return;
      for (Type type : body.getResultTypes())
        if (!type.isF32() && !type.isInteger(1))
          return;
      for (Value value : body.getOperands())
        if (value.getParentRegion() != &op.getRegion() &&
            !value.getDefiningOp<arith::ConstantOp>())
          return;
    }
    regions.push_back(op);
  });

  OpBuilder b(module.getContext());
  // Operators of the same code: the printed function, named `operator`.
  llvm::StringMap<StringAttr> unique;
  for (auto op : regions) {
    Location loc = op.getLoc();
    auto output = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t width = output.getShape().back();
    auto row = MemRefType::get({width}, b.getF32Type());
    SmallVector<bool> scalar;
    auto maps = op.getIndexingMapsArray();
    for (AffineMap map : ArrayRef<AffineMap>(maps).drop_back()) {
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      scalar.push_back(
          !(isa<AffineDimExpr>(last) &&
            cast<AffineDimExpr>(last).getPosition() == op.getNumLoops() - 1));
    }
    func::FuncOp function = createOperator(b, module, loc);
    SmallVector<Value> inputs;
    for (auto [i, isScalar] : llvm::enumerate(scalar)) {
      Value argument = function.getArgument(i);
      inputs.push_back(
          isScalar
              ? Value(arith::BitcastOp::create(
                    b, loc, b.getF32Type(),
                    arith::TruncIOp::create(b, loc, b.getI32Type(), argument)))
              : operatorMemory(b, loc, argument, row));
    }
    Value result =
        operatorMemory(b, loc, function.getArgument(scalar.size()), row);
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    auto compute = [&](Value at, int64_t lanes) {
      IRMapping mapping;
      for (auto [i, argument] :
           llvm::enumerate(op.getRegion().front().getArguments().drop_back())) {
        Value loaded;
        if (scalar[i]) {
          loaded = inputs[i];
        } else if (lanes) {
          loaded = vector::LoadOp::create(
              b, loc, VectorType::get({lanes}, b.getF32Type()), inputs[i],
              ValueRange{at});
        } else {
          loaded = memref::LoadOp::create(b, loc, inputs[i], ValueRange{at});
        }
        mapping.map(argument, loaded);
      }
      for (Operation &body : op.getRegion().front().without_terminator()) {
        for (Value operand : body.getOperands()) {
          if (mapping.contains(operand))
            continue;
          auto constant = operand.getDefiningOp<arith::ConstantOp>();
          Value value = arith::ConstantOp::create(b, loc, constant.getType(),
                                                  constant.getValue());
          mapping.map(operand, value);
        }
        if (auto erf = dyn_cast<math::ErfOp>(body)) {
          mapping.map(erf.getResult(),
                      lowerErf(b, loc, mapping.lookup(erf.getOperand())));
          continue;
        }
        bool scalarBody = llvm::all_of(body.getOperands(), [&](Value value) {
          return !isa<VectorType>(mapping.lookup(value).getType());
        });
        if (!lanes || scalarBody) {
          b.clone(body, mapping);
          continue;
        }
        OperationState state(loc, body.getName());
        for (Value operand : body.getOperands()) {
          Value value = mapping.lookup(operand);
          if (!isa<VectorType>(value.getType()))
            value = vector::BroadcastOp::create(
                b, loc, VectorType::get({lanes}, value.getType()), value);
          state.addOperands(value);
        }
        for (Type type : body.getResultTypes())
          state.addTypes(VectorType::get({lanes}, type));
        state.addAttributes(body.getAttrs());
        Operation *created = b.create(state);
        mapping.map(body.getResults(), created->getResults());
      }
      Value value =
          mapping.lookup(op.getRegion().front().getTerminator()->getOperand(0));
      if (lanes && !isa<VectorType>(value.getType()))
        value = vector::BroadcastOp::create(
            b, loc, VectorType::get({lanes}, value.getType()), value);
      if (lanes)
        vector::StoreOp::create(b, loc, value, result, ValueRange{at});
      else
        memref::StoreOp::create(b, loc, value, result, ValueRange{at});
    };
    bool hasErf = llvm::any_of(op.getRegion().front(), [](Operation &body) {
      return isa<math::ErfOp>(body);
    });
    if (hasErf) {
      auto loop = scf::ForOp::create(b, loc, index(0), index(width / 16 * 16),
                                     index(16));
      b.setInsertionPointToStart(loop.getBody());
      compute(loop.getInductionVar(), 16);
      b.setInsertionPointAfter(loop);
      for (int64_t tail = width / 16 * 16; tail < width; tail += 4)
        compute(index(tail), 4);
    } else {
      if (width >= 32) {
        auto loop = scf::ForOp::create(b, loc, index(0), index(width / 32 * 32),
                                       index(32));
        b.setInsertionPointToStart(loop.getBody());
        compute(loop.getInductionVar(), 32);
        b.setInsertionPointAfter(loop);
      }
      for (int64_t tail = width / 32 * 32; tail < width; ++tail)
        compute(index(tail), 0);
    }
    func::ReturnOp::create(b, loc);

    StringAttr name = function.getSymNameAttr();
    function.setName("operator");
    std::string code;
    llvm::raw_string_ostream printed(code);
    function.print(printed,
                   OpPrintingFlags().useLocalScope().enableDebugInfo(false));
    auto [found, inserted] = unique.try_emplace(code, name);
    if (inserted) {
      function.setName(name);
    } else {
      name = found->second;
      function.erase();
    }

    b.setInsertionPoint(op);
    SmallVector<Value> operands(op.getDpsInputs());
    SmallVector<int64_t> inputOffsets(operands.size(), 0);
    SmallVector<int64_t> inputWidths;
    for (auto [i, input] : llvm::enumerate(operands)) {
      inputWidths.push_back(
          cast<RankedTensorType>(input.getType()).getShape().back());
      auto slice = input.getDefiningOp<tensor::ExtractSliceOp>();
      if (!slice ||
          slice.getSourceType().getRank() != slice.getType().getRank() ||
          slice.getSourceType().getShape().drop_back() !=
              slice.getType().getShape().drop_back() ||
          !llvm::all_of(slice.getStaticStrides(),
                        [](int64_t n) { return n == 1; }) ||
          !llvm::all_of(slice.getStaticOffsets().drop_back(),
                        [](int64_t n) { return n == 0; }) ||
          ShapedType::isDynamic(slice.getStaticOffsets().back()))
        continue;
      inputOffsets[i] = slice.getStaticOffsets().back();
      operands[i] = slice.getSource();
    }
    auto call = ::buddy::acc::CustomOp::create(b, loc, TypeRange{output},
                                               "row_operator", operands);
    call->setAttr("operator", FlatSymbolRefAttr::get(name));
    call->setAttr("maps", b.getAffineMapArrayAttr(maps));
    call->setAttr("input_offsets", b.getDenseI64ArrayAttr(inputOffsets));
    call->setAttr("input_widths", b.getDenseI64ArrayAttr(inputWidths));
    op.getResult(0).replaceAllUsesWith(call.getResult(0));
    op.erase();
  }
  if (failed(outlineReductions(module)))
    return failure();
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!function.getName().starts_with("operator_"))
      continue;
    RewritePatternSet approximations(module.getContext());
    populateMathPolynomialApproximationPatterns(
        approximations, [](StringRef name) { return name == "math.exp"; });
    if (failed(applyPatternsGreedily(function, std::move(approximations))))
      return failure();
  }
  return success();
}
