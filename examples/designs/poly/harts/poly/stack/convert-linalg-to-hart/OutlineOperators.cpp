//===- OutlineOperators.cpp - GP domain operators of tensor regions
//-----------===//
//
// convert-linalg-to-hart first compiles the FP32 regions the GP domain of the
// Poly threads runs into operators of the module and replaces each region by a
// bbacc.custom "row_operator" that names its operator:
//
//   - a pointwise linalg.generic of static FP32 rows (the last dimension
//     at least sixteen, a multiple of four) whose inputs are rows or scalars
//     of a row (one per row or one for all), at most six inputs and three
//     vector rows;
//     MXINT8 row inputs are decoded within the same generated operator;
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

LogicalResult outlineReductions(ModuleOp module);
void preparePolyViews(ModuleOp module);

// A new operator at the end of `module`, `b` at the start of its body.
func::FuncOp createOperator(OpBuilder &b, ModuleOp module, Location loc) {
  auto space = MemRefType::get({INT64_C(1) << 40}, b.getI8Type());
  b.setInsertionPointToEnd(module.getBody());
  if (!module.lookupSymbol<memref::GlobalOp>("gp_space")) {
    auto global =
        memref::GlobalOp::create(b, loc, "gp_space", StringAttr(), space,
                                 Attribute(), false, IntegerAttr());
    global->setAttr("bbacc.gp_space", b.getUnitAttr());
  }
  int64_t number = 0;
  while (module.lookupSymbol("operator_" + std::to_string(number)))
    ++number;
  SmallVector<Type> arguments(8, b.getI64Type());
  auto function =
      func::FuncOp::create(b, loc, "operator_" + std::to_string(number),
                           b.getFunctionType(arguments, {}));
  function.setPrivate();
  function->setAttr(
      "llvm.linkage",
      LLVM::LinkageAttr::get(b.getContext(), LLVM::Linkage::Internal));
  function->setAttr("bbacc.gp_operator", b.getUnitAttr());
  function->setAttr("no_inline", b.getUnitAttr());
  function->setAttr("llvm.no_inline", b.getUnitAttr());
  b.setInsertionPointToStart(function.addEntryBlock());
  return function;
}

// The memory of `type` at the GP address `address`.
Value operatorMemory(OpBuilder &b, Location loc, Value address,
                     MemRefType type) {
  auto space = MemRefType::get({INT64_C(1) << 40}, b.getI8Type());
  Value base = memref::GetGlobalOp::create(b, loc, space, "gp_space");
  Value shift = arith::IndexCastOp::create(b, loc, b.getIndexType(), address);
  SmallVector<Value> dimensions;
  if (!type.hasStaticShape()) {
    auto function = cast<func::FuncOp>(b.getInsertionBlock()->getParentOp());
    dimensions.push_back(arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                    function.getArgument(6)));
  }
  return memref::ViewOp::create(b, loc, type, base, shift, dimensions);
}

namespace {
// A half-row rotation stays inside one GP row; DMA loads its source once.
AffineMap rowMap(AffineMap map, int64_t width) {
  if (width < 64 || width % 64)
    return map;
  AffineExpr column = getAffineDimExpr(map.getNumDims() - 1, map.getContext());
  if (map.getResults().back() != (column + width / 2) % width)
    return map;
  SmallVector<AffineExpr> results(map.getResults());
  results.back() = column;
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                        map.getContext());
}

} // namespace

// Part of convert-linalg-to-hart, before it places the graph.
LogicalResult outlineOperators(ModuleOp module) {
  preparePolyViews(module);
  SmallVector<linalg::ReduceOp> reductions;
  module.walk([&](linalg::ReduceOp op) { reductions.push_back(op); });
  IRRewriter rewriter(module.getContext());
  for (auto op : reductions) {
    rewriter.setInsertionPoint(op);
    if (failed(linalg::generalizeNamedOp(rewriter, op)))
      return op.emitError("cannot generalize reduction");
  }
  // Fuse row arithmetic without expanding contractions through reshapes:
  // convert-linalg-to-hart must still see their matmul operations.
  RewritePatternSet patterns(module.getContext());
  math::populateExpansionPatterns(patterns, {"fpowi"});
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  linalg::populateElementwiseOpsFusionPatterns(patterns, [](OpOperand
                                                                *operand) {
    Operation *producer = operand->get().getDefiningOp();
    // Broadcasting a scalar power must reuse its result, not recompute it per
    // column.
    if (auto generic = dyn_cast_or_null<linalg::GenericOp>(producer);
        generic &&
        !cast<linalg::GenericOp>(operand->getOwner())
             .getMatchingIndexingMap(operand)
             .isPermutation() &&
        llvm::any_of(generic.getRegion().front(),
                     [](Operation &op) { return isa<math::PowFOp>(op); }))
      return false;
    if (cast<linalg::GenericOp>(operand->getOwner()).getNumReductionLoops()) {
      auto square = dyn_cast_or_null<linalg::GenericOp>(producer);
      if (!square || square.getNumDpsInputs() != 1 ||
          square.getRegion().front().getOperations().size() != 2 ||
          !llvm::all_of(square.getIndexingMapsArray(),
                        [](AffineMap map) { return map.isIdentity(); }))
        return false;
      auto multiply =
          dyn_cast<arith::MulFOp>(square.getRegion().front().front());
      if (!multiply ||
          multiply.getLhs() != square.getRegion().front().getArgument(0) ||
          multiply.getRhs() != square.getRegion().front().getArgument(0))
        return false;
    }
    // Materialize gathers once; captured tensors cannot run in GP space.
    if (auto gather = dyn_cast_or_null<linalg::GenericOp>(producer))
      if (llvm::any_of(gather.getRegion().front(), [](Operation &op) {
            return isa<tensor::ExtractOp>(op);
          }))
        return false;
    return producer && llvm::all_of(producer->getUsers(), [&](Operation *user) {
             return user == operand->getOwner() || isa<tensor::DimOp>(user);
           });
  });
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();
  // Fusion forms copies and +0 of decoded values (a key head shared by a
  // group of query heads): return them to the views of the products.
  preparePolyViews(module);
  SmallVector<linalg::GenericOp> regions;
  module.walk([&](linalg::GenericOp op) {
    if (op->getParentOfType<func::FuncOp>()->hasAttr("bbacc.target"))
      return;
    if (!op.hasPureTensorSemantics() || op.getNumResults() != 1 ||
        op.getNumLoops() != op.getNumParallelLoops() ||
        op.getNumDpsInputs() == 0 || op.getNumDpsInputs() > 6)
      return;
    auto output = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!output || output.getRank() == 0 || !output.getElementType().isF32() ||
        (!output.isDynamicDim(output.getRank() - 1) &&
         (output.getShape().back() < 16 || output.getShape().back() % 4)) ||
        (output.isDynamicDim(output.getRank() - 1) &&
         op.getNumDpsInputs() > 5) ||
        !op.getIndexingMapsArray().back().isIdentity() ||
        !op.getRegion().front().getArguments().back().use_empty())
      return;
    auto maps = op.getIndexingMapsArray();
    SmallVector<std::pair<Value, AffineMap>> vectors;
    for (auto [input, originalMap] : llvm::zip_equal(
             op.getDpsInputs(), ArrayRef<AffineMap>(maps).drop_back())) {
      // Decode an arithmetic input inside its GP operator. Matrix consumers
      // keep their frontend decode so they still read packed operands.
      Value source = input;
      while (Operation *view = source.getDefiningOp()) {
        if (view->getName().getStringRef() == "frontend.mxint8_to_fp32") {
          auto type = cast<RankedTensorType>(input.getType());
          if (source != input || type.getShape().back() <= 0 ||
              type.getShape().back() != output.getShape().back() ||
              type.getShape().back() % 32 ||
              view->getAttrOfType<IntegerAttr>("axis").getInt() !=
                  type.getRank() - 1)
            return;
          break;
        }
        if (auto transpose = dyn_cast<linalg::TransposeOp>(view))
          source = transpose.getInput();
        else if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
                     tensor::ExtractSliceOp>(view))
          source = view->getOperand(0);
        else
          break;
      }
      AffineMap map = rowMap(originalMap, output.getShape().back());
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || type.getRank() == 0 ||
          (!type.getElementType().isF32() &&
           !type.getElementType().isInteger(8)) ||
          !map.isProjectedPermutation(true))
        return;
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      bool vector =
          isa<AffineDimExpr>(last) &&
          cast<AffineDimExpr>(last).getPosition() == op.getNumLoops() - 1;
      if (!vector && source.getDefiningOp() &&
          source.getDefiningOp()->getName().getStringRef() ==
              "frontend.mxint8_to_fp32")
        return;
      if (vector &&
          !llvm::is_contained(vectors, std::pair<Value, AffineMap>{input, map}))
        vectors.emplace_back(input, map);
      if (vector && type.getElementType().isInteger(8) &&
          (type.getShape().back() < 0 || type.getShape().back() % 16))
        return;
      for (AffineExpr expression : map.getResults())
        if (auto constant = dyn_cast<AffineConstantExpr>(expression);
            constant && constant.getValue() != 0)
          return;
    }
    if (vectors.size() > 3)
      return;
    for (int64_t axis = 0; axis < output.getRank(); ++axis) {
      if (!output.isDynamicDim(axis))
        continue;
      bool supplied = false;
      for (AffineMap map : ArrayRef<AffineMap>(maps).drop_back())
        for (AffineExpr expression : map.getResults())
          if (auto dim = dyn_cast<AffineDimExpr>(expression))
            supplied |= dim.getPosition() == axis;
      if (!supplied)
        return;
    }
    for (Operation &body : op.getRegion().front().without_terminator()) {
      StringRef name = body.getName().getStringRef();
      if (auto coordinate = dyn_cast<linalg::IndexOp>(body)) {
        int64_t width = output.getShape().back();
        if (coordinate.getDim() != uint64_t(output.getRank() - 1) ||
            width < 64 || width % 64 ||
            !llvm::all_of(
                coordinate->getUsers(),
                [&](Operation *user) {
                  auto compare = dyn_cast<arith::CmpIOp>(user);
                  auto threshold =
                      compare ? compare.getRhs()
                                    .getDefiningOp<arith::ConstantIndexOp>()
                              : arith::ConstantIndexOp();
                  return compare &&
                         compare.getPredicate() == arith::CmpIPredicate::ult &&
                         compare.getLhs() == coordinate.getResult() &&
                         threshold && threshold.value() == width / 2;
                }))
          return;
        continue;
      }
      if (name != "arith.extui" && name != "arith.extsi" &&
          name != "arith.sitofp" && name != "arith.bitcast" &&
          name != "arith.addi" && name != "arith.subi" &&
          name != "arith.shli" && name != "arith.shrsi" &&
          name != "arith.cmpi" && name != "arith.constant" &&
          name != "arith.addf" && name != "arith.subf" &&
          name != "arith.mulf" && name != "arith.divf" &&
          name != "arith.negf" && name != "arith.cmpf" &&
          name != "arith.select" && name != "arith.maximumf" &&
          name != "arith.minimumf" && name != "math.sqrt" &&
          name != "math.rsqrt" && name != "math.absf" && name != "math.erf" &&
          name != "math.exp" && name != "math.tanh")
        return;
      for (Type type : body.getResultTypes())
        if (!type.isF32() && !type.isInteger(1) && !type.isInteger(32))
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
    SmallVector<bool> rotated;
    for (AffineMap &map : MutableArrayRef<AffineMap>(maps).drop_back()) {
      AffineMap normalized = rowMap(map, width);
      rotated.push_back(normalized != map);
      map = normalized;
    }
    for (AffineMap map : ArrayRef<AffineMap>(maps).drop_back()) {
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      scalar.push_back(
          !(isa<AffineDimExpr>(last) &&
            cast<AffineDimExpr>(last).getPosition() == op.getNumLoops() - 1));
    }
    func::FuncOp function = createOperator(b, module, loc);
    SmallVector<Value> inputs;
    SmallVector<bool> decoded;
    for (auto [i, isScalar] : llvm::enumerate(scalar)) {
      Value argument = function.getArgument(i);
      Value source = op.getDpsInputs()[i];
      Operation *decode = source.getDefiningOp();
      bool packed = decode && decode->getName().getStringRef() ==
                                  "frontend.mxint8_to_fp32";
      decoded.push_back(packed);
      Type element = cast<RankedTensorType>(source.getType()).getElementType();
      if (isScalar) {
        Value bits = arith::TruncIOp::create(
            b, loc, element.isF32() ? b.getI32Type() : element, argument);
        inputs.push_back(element.isF32() ? Value(arith::BitcastOp::create(
                                               b, loc, element, bits))
                                         : bits);
      } else {
        inputs.push_back(operatorMemory(
            b, loc, argument,
            packed ? MemRefType::get({width + width / 32}, b.getI8Type())
                   : MemRefType::get({width}, element)));
      }
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
        Value position =
            rotated[i]
                ? Value(arith::RemUIOp::create(
                      b, loc,
                      arith::AddIOp::create(b, loc, at, index(width / 2)),
                      index(width)))
                : at;
        if (scalar[i]) {
          loaded = inputs[i];
        } else if (lanes) {
          loaded = vector::LoadOp::create(
              b, loc,
              VectorType::get(
                  {lanes},
                  cast<MemRefType>(inputs[i].getType()).getElementType()),
              inputs[i], ValueRange{position});
        } else {
          loaded =
              memref::LoadOp::create(b, loc, inputs[i], ValueRange{position});
        }
        if (decoded[i]) {
          auto i32 = [&](int64_t value) -> Value {
            return arith::ConstantIntOp::create(b, loc, value, 32);
          };
          Value scaleAt = arith::AddIOp::create(
              b, loc, index(width),
              arith::DivUIOp::create(b, loc, position, index(32)));
          Value exponent = arith::SubIOp::create(
              b, loc,
              arith::ExtUIOp::create(
                  b, loc, b.getI32Type(),
                  memref::LoadOp::create(b, loc, inputs[i],
                                         ValueRange{scaleAt})),
              i32(127));
          Value first = arith::ShRSIOp::create(b, loc, exponent, i32(1));
          Value second = arith::SubIOp::create(b, loc, exponent, first);
          loaded = arith::SIToFPOp::create(
              b, loc,
              lanes ? Type(VectorType::get({lanes}, b.getF32Type()))
                    : Type(b.getF32Type()),
              loaded);
          // Preserve the frontend's multiplication order and subnormals.
          for (Value half : {first, second}) {
            Value bits = arith::ShLIOp::create(
                b, loc, arith::AddIOp::create(b, loc, half, i32(127)), i32(23));
            Value scale;
            if (lanes) {
              // Keep the integer bit broadcast; a floating splat becomes an
              // indexed gather instead of a scalar-to-vector move.
              auto scalable =
                  VectorType::get({lanes / 2}, b.getI32Type(), {true});
              Value value =
                  LLVM::CallIntrinsicOp::create(
                      b, loc, scalable, b.getStringAttr("llvm.riscv.vmv.v.x"),
                      ValueRange{
                          LLVM::PoisonOp::create(b, loc, scalable), bits,
                          arith::ConstantIntOp::create(b, loc, lanes, 64)})
                      .getResult(0);
              value = vector::ScalableExtractOp::create(
                  b, loc, VectorType::get({lanes}, b.getI32Type()), value, 0);
              scale = arith::BitcastOp::create(
                  b, loc, VectorType::get({lanes}, b.getF32Type()), value);
            } else {
              scale = arith::BitcastOp::create(b, loc, b.getF32Type(), bits);
            }
            loaded = arith::MulFOp::create(b, loc, loaded, scale);
          }
        }
        mapping.map(argument, loaded);
      }
      for (Operation &body : op.getRegion().front().without_terminator()) {
        if (auto coordinate = dyn_cast<linalg::IndexOp>(body)) {
          mapping.map(coordinate.getResult(), at);
          continue;
        }
        for (Value operand : body.getOperands()) {
          if (mapping.contains(operand))
            continue;
          auto constant = operand.getDefiningOp<arith::ConstantOp>();
          Value value = arith::ConstantOp::create(b, loc, constant.getType(),
                                                  constant.getValue());
          mapping.map(operand, value);
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
    if (width < 0) {
      Value extent = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                function.getArgument(6));
      int64_t lanes = 32;
      Value bulk = arith::MulIOp::create(
          b, loc, arith::DivUIOp::create(b, loc, extent, index(lanes)),
          index(lanes));
      auto loop = scf::ForOp::create(b, loc, index(0), bulk, index(lanes));
      b.setInsertionPointToStart(loop.getBody());
      compute(loop.getInductionVar(), lanes);
      b.setInsertionPointAfter(loop);
      auto tail = scf::ForOp::create(b, loc, bulk, extent, index(1));
      b.setInsertionPointToStart(tail.getBody());
      compute(tail.getInductionVar(), 0);
      b.setInsertionPointAfter(tail);
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
      if (decoded[i]) {
        operands[i] = input.getDefiningOp()->getOperand(0);
        inputWidths.push_back(width + width / 32);
        continue;
      }
      inputWidths.push_back(
          cast<RankedTensorType>(input.getType()).getShape().back());
      auto slice = input.getDefiningOp<tensor::ExtractSliceOp>();
      if (!slice || !slice.getSourceType().hasStaticShape() ||
          !slice.getType().hasStaticShape() ||
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
    if (width < 0) {
      auto input = llvm::find(scalar, false);
      call->setAttr("width_input", b.getI64IntegerAttr(input - scalar.begin()));
    }
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
        approximations, [](StringRef name) {
          return name == "math.exp" || name == "math.erf" ||
                 name == "math.tanh";
        });
    if (failed(applyPatternsGreedily(function, std::move(approximations))))
      return failure();
  }
  return success();
}
