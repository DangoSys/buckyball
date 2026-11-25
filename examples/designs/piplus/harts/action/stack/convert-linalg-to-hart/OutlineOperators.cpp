//===- OutlineOperators.cpp - GP domain operators of tensor regions
//-----------===//
//
// convert-linalg-to-hart first compiles the FP32 regions the GP domain of the
// piplus threads runs into operators of the module and replaces each region by
// a bbacc.custom "row_operator" that names its operator:
//
//   - a pointwise linalg.generic of static FP32 rows (the last dimension
//     at least sixteen) whose inputs are rows or scalars
//     of a row (one per row or one for all), at most six inputs; exp and
//     tanh become polynomial approximations, a constant integer power
//     multiplications;
//   - a row sum or maximum (OutlineReductions.cpp);
//   - a transpose that keeps the rows, off the decoding of a product operand
//     (a pointwise copy of rows, generalized first), and an argument of the
//     graph inserted into a larger value (copied by a pointwise copy).
//
// Softmax over rows is not an operator of the GP domain: it becomes a
// bbacc.custom "softmax" for SoftmaxBall (OutlineSoftmax.cpp).
//
// A group of a bank holds a chunk of a row: the operator of a row wider than a
// group computes a chunk, the widest that divides the row, a multiple of
// four (attribute `chunk`); the thread runs it on each chunk of the row.
// Several whole rows go to one launch (attribute `rows`, R, the most that
// divide the axis before the row and fit a group) when every input follows
// that axis or is broadcast along it and there are no scalar inputs (the
// RoPE of the attention heads, the heads to their tokens and back): input i
// holds R or 1 rows, the operator loops over the rows, and DMA moves the R
// rows of an input at once, contiguous or apart (a transposed input).
// The groups are those of the bank of the thread of hart class `hart`
// (design.pb).
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
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;

LogicalResult outlineReductions(ModuleOp module, int64_t group);
void outlineSoftmax(ModuleOp module);
void outlineAttention(ModuleOp module);
void outlineLayerNorm(ModuleOp module);

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
  // The entry calls the operators: inlined, it would hold each of them again.
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
  return memref::ViewOp::create(b, loc, type, base, shift, ValueRange{});
}

// Part of convert-linalg-to-hart: the operators of the graph of a hart of
// class `hartClass`.
LogicalResult outlineOperators(ModuleOp module,
                               const buckyball::config::Design &design,
                               StringRef hartClass) {
  int64_t group = 0, lanes = 0;
  for (const auto &hart : design.device().harts())
    if (hart.component() == hartClass) {
      const auto &thread = design.threads(hart.thread_indices(0));
      const auto &bank = thread.mem().bank();
      group = int64_t(bank.entries()) * bank.width() / 32;
      // The elements of one vector register.
      lanes = int64_t(thread.gpdomain().v_len()) / 32;
      break;
    }
  if (!group)
    llvm::report_fatal_error(llvm::Twine("no hart of class ") + hartClass);
  // The chunk of a row of `width`: the row if it fits a group, else the
  // widest multiple of four that divides it; none if none fits.
  auto chunkOf = [&](int64_t width) -> int64_t {
    if (width <= group)
      return width;
    for (int64_t chunk = group; chunk >= 16; chunk -= 4)
      if (width % chunk == 0 && chunk % 4 == 0)
        return chunk;
    return 0;
  };
  // Softmax over rows is SoftmaxBall work, found before fusion merges its
  // pointwise steps.
  outlineLayerNorm(module);
  outlineSoftmax(module);
  // An attention keeps its scores and probabilities on its thread
  // (OutlineAttention.cpp), found before fusion merges its mask and scale.
  outlineAttention(module);
  // A transpose that keeps the rows (the heads of an attention to its
  // tokens and back) is a pointwise copy of rows the GP domain makes, not
  // a copy of the CPU; one on the way from a decoding to a product stays
  // a view the product's DMA reads, as is one on the way to the
  // quantization of an attention's operand.
  SmallVector<linalg::TransposeOp> transposes;
  module.walk([&](linalg::TransposeOp op) {
    if (op.getPermutation().back() != int64_t(op.getPermutation().size()) - 1)
      return;
    for (Operation *user = op; user->hasOneUse();) {
      user = *user->getUsers().begin();
      auto custom = dyn_cast<::buddy::acc::CustomOp>(user);
      if (custom && custom.getKind() == "attention")
        return;
      StringRef name = user->getName().getStringRef();
      if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
               tensor::ExpandShapeOp>(user) &&
          name != "frontend.fp32_to_mxint8" &&
          name != "frontend.mxint8_to_fp32")
        break;
    }
    Value value = op.getInput();
    while (Operation *view = value.getDefiningOp()) {
      if (view->getName().getStringRef() == "frontend.mxint8_to_fp32")
        return;
      if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
               tensor::ExpandShapeOp>(view))
        break;
      value = view->getOperand(0);
    }
    transposes.push_back(op);
  });
  IRRewriter rewriter(module.getContext());
  for (linalg::TransposeOp op : transposes) {
    rewriter.setInsertionPoint(op);
    if (failed(linalg::generalizeNamedOp(rewriter, op)))
      return failure();
  }
  // An argument of the graph inserted into a larger value (the prefix's
  // keys and values before a step's own) is copied by the GP domain: a
  // pointwise copy of rows, whose result the thread writes into its place.
  SmallVector<tensor::InsertSliceOp> inserts;
  module.walk([&](tensor::InsertSliceOp op) {
    auto type = op.getSourceType();
    if (isa<BlockArgument>(op.getSource()) && type.hasStaticShape() &&
        type.getElementType().isF32() && type.getRank() >= 2 &&
        type.getShape().back() >= 16)
      inserts.push_back(op);
  });
  for (tensor::InsertSliceOp op : inserts) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto type = op.getSourceType();
    Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                          type.getElementType());
    SmallVector<AffineMap> maps(
        2, rewriter.getMultiDimIdentityMap(type.getRank()));
    SmallVector<utils::IteratorType> iterators(type.getRank(),
                                               utils::IteratorType::parallel);
    auto copy = linalg::GenericOp::create(
        rewriter, loc, TypeRange{type}, ValueRange{op.getSource()},
        ValueRange{empty}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange arguments) {
          linalg::YieldOp::create(b, loc, arguments[0]);
        });
    op.getSourceMutable().assign(copy.getResult(0));
  }
  // Fuse row arithmetic without expanding contractions through reshapes:
  // convert-linalg-to-hart must still see their matmul operations. A reshape
  // between two pointwise regions (the sum of a row Linear's partials and the
  // residual it is added to) folds into them, so they fuse.
  RewritePatternSet patterns(module.getContext());
  linalg::populateElementwiseOpsFusionPatterns(
      patterns, [](OpOperand *operand) {
        Operation *producer = operand->get().getDefiningOp();
        // A fused region stays within the six inputs of a row operator.
        auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
        auto generic = dyn_cast_or_null<linalg::GenericOp>(producer);
        if (consumer && generic &&
            consumer.getNumDpsInputs() - 1 + generic.getNumDpsInputs() > 6)
          return false;
        return producer && producer->hasOneUse();
      });
  linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, [](OpOperand
                                                                     *operand) {
    auto pointwise = [](Operation *op) {
      auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
      return generic && generic.getNumLoops() == generic.getNumParallelLoops();
    };
    Operation *producer = operand->get().getDefiningOp();
    Operation *consumer = operand->getOwner();
    auto reshape = [](Operation *op) {
      return isa_and_nonnull<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          op);
    };
    if (reshape(producer))
      producer = producer->getOperand(0).getDefiningOp();
    if (reshape(consumer)) {
      if (!consumer->hasOneUse())
        return false;
      consumer = *consumer->getUsers().begin();
    }
    return pointwise(producer) && pointwise(consumer) &&
           operand->get().hasOneUse();
  });
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();
  SmallVector<linalg::GenericOp> regions;
  module.walk([&](linalg::GenericOp op) {
    // The start of a product (FuseBias.cpp's residual and bias) stays: the
    // thread loads it into the C tiles.
    if (op->hasOneUse() && isa<linalg::MatmulOp>(*op->getUsers().begin()) &&
        op->getUses().begin()->getOperandNumber() == 2)
      return;
    if (!op.hasPureTensorSemantics() || op.getNumResults() != 1 ||
        op.getNumLoops() != op.getNumParallelLoops() ||
        op.getNumDpsInputs() > 6)
      return;
    auto output = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!output || !output.hasStaticShape() ||
        !output.getElementType().isF32() || output.getRank() == 0 ||
        output.getShape().back() < 16 ||
        !op.getIndexingMapsArray().back().isIdentity() ||
        !op.getRegion().front().getArguments().back().use_empty() ||
        !chunkOf(output.getShape().back()))
      return;
    auto maps = op.getIndexingMapsArray();
    for (auto [input, map] : llvm::zip_equal(
             op.getDpsInputs(), ArrayRef<AffineMap>(maps).drop_back())) {
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || !type.hasStaticShape() ||
          !(type.getElementType().isF32() ||
            type.getElementType().isInteger(1)) ||
          type.getRank() == 0 || !map.isProjectedPermutation(true))
        return;
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      bool vector =
          isa<AffineDimExpr>(last) &&
          cast<AffineDimExpr>(last).getPosition() == op.getNumLoops() - 1;
      if (!vector &&
          (type.getShape().back() != 1 || !type.getElementType().isF32()))
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
          name != "math.exp" && name != "math.tanh" && name != "math.fpowi")
        return;
      for (Type type : body.getResultTypes())
        if (!type.isF32() && !type.isInteger(1))
          return;
      if (auto power = dyn_cast<math::FPowIOp>(body);
          power && !power.getRhs().getDefiningOp<arith::ConstantIntOp>())
        return;
      for (Value value : body.getOperands())
        if (value.getParentRegion() != &op.getRegion() &&
            !value.getDefiningOp<arith::ConstantOp>())
          return;
    }
    Value yielded = op.getRegion().front().getTerminator()->getOperand(0);
    if (yielded.getParentRegion() != &op.getRegion() &&
        !yielded.getDefiningOp<arith::ConstantOp>())
      return;
    regions.push_back(op);
  });

  OpBuilder b(module.getContext());
  // Operators of the same code: the printed function, named `operator`.
  llvm::StringMap<StringAttr> unique;
  // A mask row is a row of bytes: the byte tensor of each mask, made once
  // after it.
  for (auto op : regions) {
    Location loc = op.getLoc();
    auto output = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t chunk = chunkOf(output.getShape().back());
    int64_t width = chunk;
    SmallVector<bool> scalar;
    auto maps = op.getIndexingMapsArray();
    for (AffineMap map : ArrayRef<AffineMap>(maps).drop_back()) {
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      scalar.push_back(
          !(isa<AffineDimExpr>(last) &&
            cast<AffineDimExpr>(last).getPosition() == op.getNumLoops() - 1));
    }
    // The rows of a launch: the axis before the row, `along` the inputs
    // that follow it (the others broadcast along it). The R rows of an input
    // are contiguous or rows of a power of two 16-byte rows (block DMA).
    int64_t rank = output.getRank(), rowsOf = 1;
    SmallVector<bool> along(scalar.size(), false);
    if (rank >= 2 && chunk == output.getShape().back() && width % 4 == 0 &&
        llvm::none_of(scalar, [](bool s) { return s; })) {
      int64_t rows = output.getDimSize(rank - 2);
      // The most rows that divide the axis and fit a group.
      int64_t most = 0;
      for (int64_t count = std::min(rows, group / width); count >= 2 && !most;
           --count)
        if (rows % count == 0)
          most = count;
      bool fits = most >= 2;
      for (auto [i, map] :
           llvm::enumerate(ArrayRef<AffineMap>(maps).drop_back())) {
        auto inputType = cast<RankedTensorType>(op.getDpsInputs()[i].getType());
        bool uses = llvm::any_of(map.getResults(), [&](AffineExpr e) {
          return e.isFunctionOfDim(rank - 2);
        });
        // The axis among the input's before its row: right before a whole
        // row its rows are contiguous, elsewhere (a transpose) or before a
        // slice of a row they are apart.
        int64_t at = -1;
        for (int64_t result = 0; result + 1 < map.getNumResults(); ++result)
          if (map.getResult(result) ==
              getAffineDimExpr(rank - 2, op.getContext()))
            at = result;
        int64_t blocks = width / 4;
        bool sliced =
            bool(op.getDpsInputs()[i].getDefiningOp<tensor::ExtractSliceOp>());
        bool moves = (at == map.getNumResults() - 2 && !sliced &&
                      inputType.getShape().back() == width) ||
                     (llvm::isPowerOf2_64(blocks) && blocks <= 16);
        fits &= !getElementTypeOrSelf(inputType).isInteger(1) &&
                (!uses || (at >= 0 && moves));
        along[i] = uses;
      }
      if (fits)
        rowsOf = most;
    }
    auto row = MemRefType::get({rowsOf, width}, b.getF32Type());
    func::FuncOp function = createOperator(b, module, loc);
    SmallVector<Value> inputs;
    SmallVector<bool> mask;
    for (auto [i, isScalar] : llvm::enumerate(scalar)) {
      Value argument = function.getArgument(i);
      mask.push_back(
          getElementTypeOrSelf(op.getDpsInputs()[i].getType()).isInteger(1));
      inputs.push_back(
          isScalar
              ? Value(arith::BitcastOp::create(
                    b, loc, b.getF32Type(),
                    arith::TruncIOp::create(b, loc, b.getI32Type(), argument)))
          : mask[i]
              ? operatorMemory(b, loc, argument,
                               MemRefType::get({1, width}, b.getI8Type()))
              : operatorMemory(b, loc, argument,
                               MemRefType::get({along[i] ? rowsOf : 1, width},
                                               b.getF32Type())));
    }
    Value result =
        operatorMemory(b, loc, function.getArgument(scalar.size()), row);
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    Value rowAt;
    auto compute = [&](Value at, int64_t lanes) {
      IRMapping mapping;
      for (auto [i, argument] :
           llvm::enumerate(op.getRegion().front().getArguments().drop_back())) {
        Value loaded;
        if (scalar[i]) {
          loaded = inputs[i];
        } else if (lanes) {
          loaded = vector::LoadOp::create(
              b, loc,
              VectorType::get({lanes}, mask[i] ? Type(b.getI8Type())
                                               : Type(b.getF32Type())),
              inputs[i], ValueRange{along[i] ? rowAt : index(0), at});
          if (mask[i])
            loaded = arith::TruncIOp::create(
                b, loc, VectorType::get({lanes}, b.getI1Type()), loaded);
        } else {
          loaded = memref::LoadOp::create(
              b, loc, inputs[i], ValueRange{along[i] ? rowAt : index(0), at});
          if (mask[i])
            loaded = arith::TruncIOp::create(b, loc, b.getI1Type(), loaded);
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
        // A constant power: square and multiply.
        if (auto power = dyn_cast<math::FPowIOp>(body)) {
          Value x = mapping.lookup(power.getLhs());
          int64_t n =
              power.getRhs().getDefiningOp<arith::ConstantIntOp>().value();
          if (n < 1)
            llvm::report_fatal_error("convert-linalg-to-hart: power below one");
          Value product;
          for (Value square = x; n; n >>= 1) {
            if (n & 1)
              product =
                  product
                      ? Value(arith::MulFOp::create(b, loc, product, square))
                      : square;
            if (n > 1)
              square = arith::MulFOp::create(b, loc, square, square);
          }
          mapping.map(power.getResult(), product);
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
      // The row: a value of the region or a constant.
      Value yielded = op.getRegion().front().getTerminator()->getOperand(0);
      Value value = mapping.lookupOrNull(yielded);
      if (!value) {
        auto constant = yielded.getDefiningOp<arith::ConstantOp>();
        value = arith::ConstantOp::create(b, loc, constant.getType(),
                                          constant.getValue());
      }
      if (lanes && !isa<VectorType>(value.getType()))
        value = vector::BroadcastOp::create(
            b, loc, VectorType::get({lanes}, value.getType()), value);
      if (lanes)
        vector::StoreOp::create(b, loc, value, result, ValueRange{rowAt, at});
      else
        memref::StoreOp::create(b, loc, value, result, ValueRange{rowAt, at});
    };
    auto rowLoop =
        scf::ForOp::create(b, loc, index(0), index(rowsOf), index(1));
    b.setInsertionPointToStart(rowLoop.getBody());
    rowAt = rowLoop.getInductionVar();
    // Pairs of registers (LMUL 2: the splatted constants of a polynomial
    // and its temporaries stay in sixteen groups without spills), then single
    // registers, then elements.
    int64_t wide = lanes * 2, end = width / wide * wide;
    if (end) {
      auto loop = scf::ForOp::create(b, loc, index(0), index(end), index(wide));
      b.setInsertionPointToStart(loop.getBody());
      compute(loop.getInductionVar(), wide);
      b.setInsertionPointAfter(loop);
    }
    if (width - end >= lanes) {
      auto loop = scf::ForOp::create(
          b, loc, index(end), index(width / lanes * lanes), index(lanes));
      b.setInsertionPointToStart(loop.getBody());
      compute(loop.getInductionVar(), lanes);
      b.setInsertionPointAfter(loop);
    }
    if (width % lanes) {
      auto tail = scf::ForOp::create(b, loc, index(width / lanes * lanes),
                                     index(width), index(1));
      b.setInsertionPointToStart(tail.getBody());
      compute(tail.getInductionVar(), 0);
      b.setInsertionPointAfter(tail);
    }
    b.setInsertionPointAfter(rowLoop);
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

    // A mask is read as it is in memory, a byte an element.
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
    call->setAttr("chunk", b.getI64IntegerAttr(chunk));
    call->setAttr("rows", b.getI64IntegerAttr(rowsOf));
    op.getResult(0).replaceAllUsesWith(call.getResult(0));
    op.erase();
  }
  if (failed(outlineReductions(module, group)))
    return failure();
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!function.getName().starts_with("operator_"))
      continue;
    RewritePatternSet approximations(module.getContext());
    populateMathPolynomialApproximationPatterns(
        approximations, [](StringRef name) {
          return name == "math.exp" || name == "math.tanh";
        });
    if (failed(applyPatternsGreedily(function, std::move(approximations))))
      return failure();
  }
  return success();
}
