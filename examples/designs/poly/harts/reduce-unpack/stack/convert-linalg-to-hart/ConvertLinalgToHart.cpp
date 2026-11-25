//===- ConvertLinalgToHart.cpp - A graph on the poly threads --------------===//
//
// convert-linalg-to-hart places the work of a segment graph that the hardware
// threads of a poly hart can do on those threads; the rest stays on the hart
// CPU. A poly hart has attention and ffn threads (design.pb, by component),
// each with MXINT8 quantization and matrix Balls. A thread takes every
// fp32_to_mxint8 and every matrix product of decoded MXINT8 operands.
// Decoding and layout views stay with a product so its Ball reads packed data.
// CPU consumers of those values get their own decoding/layout expressions.
//
// The thread follows the semantics poly's threads were made for. A graph
// that holds the attention of a layer, products batched over the heads
// (linalg.batch_matmul: scores and context), is an attention graph: its
// projections, scores, context and output projection run on the attention
// threads. Any other graph (the MLP: gate, up and down projections) runs on
// the ffn threads.
//
// The placed operations that read each other's values form one subgraph, a
// function <function>_ht<n> of tensors: the unit a thread runs. Constants,
// empty tensors and fills of them are copied into the subgraph; every other
// value it reads is an input, transferred to its thread once, before the
// first launch that reads it. Each of its values another operation reads is
// a result.
//
// The schedule: a subgraph is launched as early as its inputs allow, right
// after the last value of the CPU it reads (or at the start of the graph),
// and each result is transferred back right before the first operation of
// the CPU that reads it, so the CPU goes on while the thread runs and the
// subgraphs that read no value of each other run at the same time. The
// subgraphs of a graph take the threads of its kind in turn, in the order of
// their launches: the gate and up projections of an MLP, which read the same
// activations, start together on ffn0 and ffn1; the query, key and value
// projections of an attention on attention0 and attention1.
//
// A size may be dynamic (the token count of a prefill). The CPU computes
// every dynamic size it needs from values it has: a size of a value of a
// subgraph is that of the operand the operation keeps it from (the frontend
// operations off their axis, the padded and packed size along it, a linalg
// operation from its init, a reshape from its output shape), down to a value
// of the CPU; a subgraph computes its own from its inputs.
//
// Two subgraphs exchange no value directly: an operation that reads the
// value of a placed operation is either placed, and then in the same
// subgraph, or on the CPU. In poly's graphs every value between two
// subgraphs passes through an operation of the CPU (RoPE, softmax, SiLU·up,
// the residual of the two-term quantization), so no value passes from one
// thread to another.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Hart/HartOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::hart;

LogicalResult outlineOperators(ModuleOp module);

namespace {
bool isQuantize(Operation *op) {
  return op->getName().getStringRef() == "frontend.fp32_to_mxint8";
}

bool isDecode(Operation *op) {
  return op->getName().getStringRef() == "frontend.mxint8_to_fp32";
}

bool isRowOperator(Operation *op) {
  auto custom = dyn_cast_or_null<::buddy::acc::CustomOp>(op);
  return custom && custom.getKind() == "row_operator";
}

// The operations from an operand of a product back to the mxint8_to_fp32
// that decodes it; none if the operand is not decoded MXINT8.
SmallVector<Operation *> decoding(Value value) {
  SmallVector<Operation *> chain;
  while (Operation *op = value.getDefiningOp()) {
    chain.push_back(op);
    if (isDecode(op)) {
      // A weight padded to the Ball layout is decoded through its slices.
      for (Value source = op->getOperand(0);
           auto slice = source.getDefiningOp<tensor::ExtractSliceOp>();
           source = slice.getSource())
        chain.push_back(slice);
      return chain;
    }
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op))
      value = transpose.getInput();
    else if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
             slice &&
             slice.getType().getRank() == slice.getSourceType().getRank())
      value = slice.getSource();
    else if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op))
      value = op->getOperand(0);
    else
      break;
  }
  return {};
}

bool positiveZero(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    value = fill.getInputs().front();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto scalar = dyn_cast<FloatAttr>(constant.getValue()))
    return scalar.getValue().isZero() && !scalar.getValue().isNegative();
  if (auto tensor = dyn_cast<DenseFPElementsAttr>(constant.getValue()))
    return tensor.isSplat() && tensor.getSplatValue<APFloat>().isZero() &&
           !tensor.getSplatValue<APFloat>().isNegative();
  return false;
}

// Keep layout views visible to matmul. Decoded MXINT8 needs no +0; for
// ordinary FP32, move +0 before a transpose without dropping its arithmetic.
void foldZeroAddViews(Block &body) {
  for (auto op : llvm::make_early_inc_range(body.getOps<linalg::GenericOp>())) {
    if (op.getNumResults() != 1 || op.getNumDpsInits() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops())
      continue;
    Block &region = op.getRegion().front();
    if (std::distance(region.begin(), region.end()) != 2)
      continue;
    auto add = dyn_cast<arith::AddFOp>(region.front());
    if (!add ||
        cast<linalg::YieldOp>(region.back()).getValues()[0] != add.getResult())
      continue;
    auto maps = op.getIndexingMapsArray();
    if (!maps.back().isIdentity())
      continue;
    for (unsigned side = 0; side < 2; ++side) {
      auto argument = dyn_cast<BlockArgument>(add.getOperand(side));
      if (!argument || argument.getOwner() != &region ||
          argument.getArgNumber() >= op.getNumDpsInputs())
        continue;
      Value input = op.getInputs()[argument.getArgNumber()];
      if (input.getType() != op.getResult(0).getType())
        continue;
      Value zero = add.getOperand(1 - side);
      if (auto arg = dyn_cast<BlockArgument>(zero)) {
        if (arg.getOwner() != &region ||
            arg.getArgNumber() >= op.getNumDpsInputs())
          continue;
        zero = op.getInputs()[arg.getArgNumber()];
      }
      if (!positiveZero(zero))
        continue;
      auto type = cast<RankedTensorType>(input.getType());
      bool identity = true;
      for (auto [axis, expr] :
           llvm::enumerate(maps[argument.getArgNumber()].getResults())) {
        auto dim = dyn_cast<AffineDimExpr>(expr);
        auto constant = dyn_cast<AffineConstantExpr>(expr);
        identity &= (dim && dim.getPosition() == axis) ||
                    (constant && constant.getValue() == 0 &&
                     type.getDimSize(axis) == 1);
      }
      if (!identity)
        continue;
      if (decoding(input).empty()) {
        auto transpose = input.getDefiningOp<linalg::TransposeOp>();
        if (!transpose || !input.hasOneUse())
          continue;
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Value source = transpose.getInput();
        auto sourceType = cast<RankedTensorType>(source.getType());
        SmallVector<Value> sizes;
        for (int64_t axis = 0; axis < sourceType.getRank(); ++axis)
          if (sourceType.isDynamicDim(axis))
            sizes.push_back(tensor::DimOp::create(builder, loc, source, axis));
        Value empty =
            tensor::EmptyOp::create(builder, loc, sourceType.getShape(),
                                    sourceType.getElementType(), sizes);
        Value zero = arith::ConstantOp::create(
            builder, loc,
            builder.getFloatAttr(sourceType.getElementType(), 0.0));
        AffineMap map = builder.getMultiDimIdentityMap(sourceType.getRank());
        auto normalized = linalg::GenericOp::create(
            builder, loc, TypeRange{sourceType}, ValueRange{source},
            ValueRange{empty}, ArrayRef<AffineMap>{map, map},
            SmallVector<utils::IteratorType>(sourceType.getRank(),
                                             utils::IteratorType::parallel),
            [&](OpBuilder &b, Location at, ValueRange args) {
              Value value =
                  arith::AddFOp::create(b, at, side == 0 ? args[0] : zero,
                                        side == 0 ? zero : args[0]);
              linalg::YieldOp::create(b, at, value);
            });
        Value moved = linalg::TransposeOp::create(
                          builder, loc, normalized.getResult(0),
                          op.getDpsInits()[0], transpose.getPermutation())
                          ->getResult(0);
        op.getResult(0).replaceAllUsesWith(moved);
        op.erase();
        transpose.erase();
        break;
      }
      op.getResult(0).replaceAllUsesWith(input);
      op.erase();
      break;
    }
  }
}

// Constants, empty tensors, fills of them and sizes: a subgraph makes its
// own.
bool isLocal(Operation *op) {
  if (isa<arith::ConstantOp, affine::AffineApplyOp, tensor::EmptyOp,
          tensor::DimOp>(op))
    return true;
  if (op->getDialect()->getNamespace() == "arith" &&
      llvm::all_of(op->getResultTypes(),
                   [](Type type) { return type.isIndex(); }))
    return true;
  if (isa<linalg::FillOp>(op))
    return llvm::all_of(op->getOperands(), [](Value value) {
      Operation *def = value.getDefiningOp();
      return def && isLocal(def);
    });
  return false;
}

// The values an operation reads: its operands and the values its regions
// read from outside.
SetVector<Value> readsOf(Operation *op) {
  SetVector<Value> values(op->operand_begin(), op->operand_end());
  getUsedValuesDefinedAbove(op->getRegions(), values);
  return values;
}

// Computes the sizes of values before `point`, an operation of `body`.
// The CPU computes them: a value of a placed operation is not one it has.
struct Sizes {
  Block &body;
  Operation *point;
  const DenseSet<Operation *> &placed;

  bool before(Value value) {
    Operation *def = value.getDefiningOp();
    return !def || (!placed.contains(def) &&
                    body.findAncestorOpInBlock(*def)->isBeforeInBlock(point));
  }

  // An index value, from those before the point.
  Value index(OpBuilder &builder, Location loc, Value value) {
    if (before(value))
      return value;
    Operation *def = value.getDefiningOp();
    if (auto dim = dyn_cast<tensor::DimOp>(def))
      return size(builder, loc, dim.getSource(), *dim.getConstantIndex());
    if (def->getDialect()->getNamespace() != "arith" &&
        !isa<affine::AffineApplyOp>(def)) {
      def->emitError("a size computed by an unknown operation");
      llvm::report_fatal_error(
          "convert-linalg-to-hart: a size it cannot compute");
    }
    IRMapping mapping;
    for (Value operand : def->getOperands())
      mapping.map(operand, index(builder, loc, operand));
    return builder.clone(*def, mapping)->getResult(0);
  }

  Value index(OpBuilder &builder, Location loc, OpFoldResult size) {
    if (auto value = dyn_cast<Value>(size))
      return index(builder, loc, value);
    return arith::ConstantIndexOp::create(
        builder, loc, cast<IntegerAttr>(cast<Attribute>(size)).getInt());
  }

  // Size `dim` of `value`.
  Value size(OpBuilder &builder, Location loc, Value value, int64_t dim) {
    auto type = cast<RankedTensorType>(value.getType());
    if (!type.isDynamicDim(dim))
      return arith::ConstantIndexOp::create(builder, loc, type.getDimSize(dim));
    if (before(value))
      return tensor::DimOp::create(builder, loc, value, dim);
    Operation *def = value.getDefiningOp();
    if (isRowOperator(def)) {
      auto maps = def->getAttrOfType<ArrayAttr>("maps");
      for (auto [input, map] :
           llvm::zip_equal(def->getOperands(),
                           ArrayRef<Attribute>(maps.getValue()).drop_back()))
        for (auto [axis, expression] :
             llvm::enumerate(cast<AffineMapAttr>(map).getValue().getResults()))
          if (auto source = dyn_cast<AffineDimExpr>(expression);
              source && source.getPosition() == dim)
            return size(builder, loc, input, axis);
    }
    if (isQuantize(def) || isDecode(def)) {
      Value source = size(builder, loc, def->getOperand(0), dim);
      if (dim != cast<IntegerAttr>(def->getAttr("axis")).getInt())
        return source;
      // Along the MXINT8 axis: K values zero-padded to Kp = ceil(K / 32) * 32
      // pack into Kp + Kp / 32 bytes, which decode to the Kp values.
      Value width = arith::ConstantIndexOp::create(builder, loc, 32);
      if (isDecode(def))
        return arith::MulIOp::create(
            builder, loc,
            arith::DivUIOp::create(
                builder, loc, source,
                arith::ConstantIndexOp::create(builder, loc, 33)),
            width);
      Value blocks = arith::CeilDivUIOp::create(builder, loc, source, width);
      return arith::AddIOp::create(
          builder, loc, arith::MulIOp::create(builder, loc, blocks, width),
          blocks);
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(def))
      return size(builder, loc,
                  dps.getTiedOpOperand(cast<OpResult>(value))->get(), dim);
    if (auto empty = dyn_cast<tensor::EmptyOp>(def))
      return index(builder, loc, empty.getDynamicSize(dim));
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def))
      return index(builder, loc, expand.getMixedOutputShape()[dim]);
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      Value product = arith::ConstantIndexOp::create(builder, loc, 1);
      auto groups = collapse.getReassociationIndices();
      for (int64_t source : groups[dim])
        product = arith::MulIOp::create(
            builder, loc, product,
            size(builder, loc, collapse.getSrc(), source));
      return product;
    }
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      llvm::SmallBitVector dropped = slice.getDroppedDims();
      for (int64_t source = 0, kept = 0;; ++source)
        if (!dropped.test(source) && kept++ == dim)
          return index(builder, loc, slice.getMixedSizes()[source]);
    }
    def->emitError("a dynamic size the CPU cannot compute");
    llvm::report_fatal_error(
        "convert-linalg-to-hart: a size it cannot compute");
  }

  // The dynamic sizes of `value`, for an empty tensor of its type.
  SmallVector<Value> dynamic(OpBuilder &builder, Location loc, Value value) {
    SmallVector<Value> sizes;
    auto type = cast<RankedTensorType>(value.getType());
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      if (type.isDynamicDim(dim))
        sizes.push_back(size(builder, loc, value, dim));
    return sizes;
  }
};

// Independent row ranges are separate jobs of the local thread pool.
void partitionRows(func::FuncOp function, int64_t count) {
  Block &body = function.getBody().front();
  DenseSet<Operation *> shapeOps;
  for (Operation &op : body)
    if (isDecode(&op) || isQuantize(&op) || isRowOperator(&op) ||
        isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
      shapeOps.insert(&op);
  int64_t group = 0;
  for (Operation *op : llvm::to_vector(
           llvm::map_range(body, [](Operation &op) { return &op; }))) {
    bool matrix = isa<linalg::MatmulOp>(op);
    if (!matrix && !isRowOperator(op))
      continue;
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    if (type.getRank() < 2 || (matrix && type.isDynamicDim(1)))
      continue;
    int64_t axis = matrix ? 0 : type.getRank() - 2;
    if (!type.isDynamicDim(axis) &&
        type.getDimSize(axis) < count * (matrix ? 16 : 1))
      continue;
    auto fill = matrix ? op->getOperand(2).getDefiningOp<linalg::FillOp>()
                       : linalg::FillOp();
    if (matrix && !fill)
      continue;
    OpBuilder b(op);
    Location loc = op->getLoc();
    Sizes shapes{body, op, shapeOps};
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    auto extent = [&](Value value, int64_t dim) -> Value {
      return shapes.size(b, loc, value, dim);
    };
    Value rows = extent(op->getResult(0), axis);
    Value units =
        matrix ? Value(arith::CeilDivUIOp::create(b, loc, rows, index(16)))
               : rows;
    SmallVector<Value> bounds{index(0)};
    for (int64_t part = 1; part < count; ++part) {
      Value boundary = arith::DivUIOp::create(
          b, loc, arith::MulIOp::create(b, loc, units, index(part)),
          index(count));
      if (matrix)
        boundary = arith::MinUIOp::create(
            b, loc, rows, arith::MulIOp::create(b, loc, boundary, index(16)));
      bounds.push_back(boundary);
    }
    bounds.push_back(rows);
    SmallVector<Value> parts, lengths;
    StringAttr identity = b.getStringAttr(
        (function.getName() + "_rows" + llvm::Twine(group++)).str());
    for (int64_t part = 0; part < count; ++part) {
      Value length =
          arith::SubIOp::create(b, loc, bounds[part + 1], bounds[part]);
      lengths.push_back(length);
      auto slice = [&](Value input, Value original, int64_t dim) -> Value {
        auto source = cast<RankedTensorType>(input.getType());
        SmallVector<OpFoldResult> offsets(source.getRank(), b.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes,
            strides(source.getRank(), b.getIndexAttr(1));
        for (int64_t d = 0; d < source.getRank(); ++d)
          sizes.push_back(
              d == dim ? OpFoldResult(length)
              : source.isDynamicDim(d)
                  ? OpFoldResult(extent(original, d))
                  : OpFoldResult(b.getIndexAttr(source.getDimSize(d))));
        offsets[dim] = bounds[part];
        return tensor::ExtractSliceOp::create(b, loc, input, offsets, sizes,
                                              strides);
      };
      IRMapping mapping;
      SmallVector<int64_t> shape(type.getShape());
      shape[axis] = ShapedType::kDynamic;
      auto partType = RankedTensorType::get(shape, type.getElementType());
      if (matrix) {
        for (unsigned operand = 0; operand < 2; ++operand) {
          auto chain = decoding(op->getOperand(operand));
          for (Operation *view : llvm::reverse(chain))
            b.clone(*view, mapping);
          if (operand == 0) {
            Value input = mapping.lookup(op->getOperand(0));
            mapping.map(op->getOperand(0), slice(input, op->getOperand(0), 0));
            for (Operation *view : chain)
              if (isDecode(view))
                if (auto quant = view->getOperand(0).getDefiningOp();
                    quant && isQuantize(quant))
                  quant->setAttr("hart.row_producer", b.getUnitAttr());
          }
        }
        Value empty = tensor::EmptyOp::create(
            b, loc, shape, type.getElementType(), ValueRange{length});
        Value initialized = linalg::FillOp::create(b, loc, fill.getDpsInputs(),
                                                   ValueRange{empty})
                                .getResult(0);
        mapping.map(op->getOperand(2), initialized);
      } else {
        auto maps = op->getAttrOfType<ArrayAttr>("maps");
        for (auto [input, attr] :
             llvm::zip_equal(op->getOperands(), maps.getValue().drop_back())) {
          auto map = cast<AffineMapAttr>(attr).getValue();
          for (auto [dim, expression] : llvm::enumerate(map.getResults()))
            if (auto d = dyn_cast<AffineDimExpr>(expression);
                d && d.getPosition() == axis)
              mapping.map(input, slice(input, input, dim));
        }
      }
      Operation *task = b.clone(*op, mapping);
      task->getResult(0).setType(partType);
      task->setAttr("hart.parallel_group", identity);
      task->setAttr("hart.parallel_part", b.getI64IntegerAttr(part));
      parts.push_back(task->getResult(0));
    }
    SmallVector<Value> dynamic;
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      if (type.isDynamicDim(dim))
        dynamic.push_back(extent(op->getResult(0), dim));
    Value result = tensor::EmptyOp::create(b, loc, type.getShape(),
                                           type.getElementType(), dynamic);
    for (int64_t part = 0; part < count; ++part) {
      SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes,
          strides(type.getRank(), b.getIndexAttr(1));
      offsets[axis] = bounds[part];
      for (int64_t dim = 0; dim < type.getRank(); ++dim)
        sizes.push_back(
            dim == axis ? OpFoldResult(lengths[part])
            : type.isDynamicDim(dim)
                ? OpFoldResult(extent(op->getResult(0), dim))
                : OpFoldResult(b.getIndexAttr(type.getDimSize(dim))));
      result = tensor::InsertSliceOp::create(b, loc, parts[part], result,
                                             offsets, sizes, strides);
    }
    op->getResult(0).replaceAllUsesWith(result);
    shapeOps.erase(op);
    op->erase();
  }
  RewritePatternSet cleanup(function.getContext());
  if (failed(applyPatternsGreedily(function, std::move(cleanup))))
    llvm::report_fatal_error("Poly row partition cleanup failed");
}

class ConvertLinalgToHartPass
    : public PassWrapper<ConvertLinalgToHartPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToHartPass)

  ConvertLinalgToHartPass() = default;
  ConvertLinalgToHartPass(const ConvertLinalgToHartPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "convert-linalg-to-hart"; }
  StringRef getDescription() const final {
    return "Place the MXINT8 quantizations and matrix products of a graph "
           "on the attention or ffn threads of its poly hart";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        ::buddy::acc::AccDialect, bufferization::BufferizationDialect,
        HartDialect, arith::ArithDialect, func::FuncDialect, math::MathDialect,
        memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
        vector::VectorDialect, LLVM::LLVMDialect>();
  }

  Option<std::string> designPb{*this, "design",
                               llvm::cl::desc("design.pb of the device")};
  Option<std::string> hartClass{
      *this, "hart", llvm::cl::desc("class of the hart the graph runs on")};

  void runOnOperation() override {
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    if (!buffer)
      llvm::report_fatal_error(llvm::Twine("cannot read ") + designPb);
    buckyball::config::Design design;
    if (!design.ParseFromArray((*buffer)->getBufferStart(),
                               (*buffer)->getBufferSize()))
      llvm::report_fatal_error(llvm::Twine("cannot parse ") + designPb);
    const buckyball::config::HartPlacement *placement = nullptr;
    for (const auto &hart : design.device().harts())
      if (hart.component() == hartClass) {
        placement = &hart;
        break;
      }
    if (!placement)
      llvm::report_fatal_error(llvm::Twine("no hart of class ") + hartClass);
    if (placement->thread_indices().empty()) {
      for (auto function : getOperation().getOps<func::FuncOp>()) {
        if (function.isExternal() || function->hasAttr("bbacc.gp_operator"))
          continue;
        function->setAttr(
            "bbacc.target",
            StringAttr::get(&getContext(), placement->component()));
        function->setAttr("hart.program_bank",
                          IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                           placement->virtual_bank_count()));
      }
      return;
    }
    if (failed(outlineOperators(getOperation())))
      return signalPassFailure();
    // The threads of each kind, by their index in the hart.
    llvm::StringMap<SmallVector<int64_t>> kinds;
    for (auto [local, index] : llvm::enumerate(placement->thread_indices())) {
      const auto &thread = design.threads(index);
      llvm::StringSet<> balls;
      for (const auto &ball : thread.balldomain().balls())
        balls.insert(ball.component());
      if (!balls.contains("mxint8quantize") || !balls.contains("mxint8matmul"))
        llvm::report_fatal_error(llvm::Twine("thread ") + thread.role() +
                                 " of hart class " + hartClass +
                                 " lacks mxint8quantize or mxint8matmul");
      kinds[thread.pkg()].push_back(local);
    }
    if (!kinds.contains("attention") || !kinds.contains("ffn"))
      llvm::report_fatal_error(llvm::Twine("hart class ") + hartClass +
                               " has no attention or no ffn thread");

    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    // Resolve shape-only uses before selecting computations for threads.
    RewritePatternSet shapes(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(shapes);
    if (failed(applyPatternsGreedily(module, std::move(shapes))))
      return signalPassFailure();
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal() || function->hasAttr("bbacc.gp_operator"))
        continue;
      Block &body = function.getBody().front();
      foldZeroAddViews(body);
      bool batched = llvm::any_of(
          body, [](Operation &op) { return isa<linalg::BatchMatmulOp>(op); });
      partitionRows(function, kinds[batched ? "attention" : "ffn"].size());
      DenseSet<Operation *> placed;
      bool attention = false;
      for (Operation &op : body) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        attention |= isa<linalg::BatchMatmulOp>(op);
        placed.insert(&op);
        SmallVector<Operation *> lhs = decoding(op.getOperand(0)),
                                 rhs = decoding(op.getOperand(1));
        if (lhs.empty() || rhs.empty()) {
          op.emitError("poly matrix products require MXINT8 operands");
          return signalPassFailure();
        }
        placed.insert(lhs.begin(), lhs.end());
        placed.insert(rhs.begin(), rhs.end());
      }
      for (Operation &op : body)
        if ((isQuantize(&op) || isRowOperator(&op)) && !op.use_empty())
          placed.insert(&op);
      // Reify shape-only views from CPU inputs, without moving data views
      // of a product away from its hart subgraph destination.
      SmallVector<Operation *> shapeViews;
      for (Operation &op : body)
        if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op) &&
            placed.contains(op.getOperand(0).getDefiningOp()) &&
            llvm::all_of(op.getUsers(), [](Operation *user) {
              return isa<tensor::DimOp>(user);
            })) {
          placed.insert(&op);
          shapeViews.push_back(&op);
        }
      ArrayRef<int64_t> threads = kinds[attention ? "attention" : "ffn"];

      // The sizes the CPU reads of values a thread computes: from the values
      // they keep them from.
      SmallVector<tensor::DimOp> dims;
      for (auto dim : body.getOps<tensor::DimOp>())
        if (placed.contains(dim.getSource().getDefiningOp()))
          dims.push_back(dim);
      for (tensor::DimOp dim : dims) {
        OpBuilder builder(dim);
        Sizes sizes{body, dim, placed};
        dim.replaceAllUsesWith(sizes.size(
            builder, dim.getLoc(), dim.getSource(), *dim.getConstantIndex()));
        dim.erase();
      }
      for (Operation *view : llvm::reverse(shapeViews)) {
        placed.erase(view);
        view->erase();
      }

      // Keep a private decoding/layout chain at each product when the CPU
      // also reads it. Thread boundaries carry the packed input, not FP32
      // views.
      DenseSet<Operation *> sharedViews;
      for (Operation &op : llvm::make_early_inc_range(body)) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        for (unsigned index = 0; index < 2; ++index) {
          auto chain = decoding(op.getOperand(index));
          bool shared = llvm::any_of(chain, [&](Operation *view) {
            return llvm::any_of(view->getUsers(), [&](Operation *user) {
              return !placed.contains(body.findAncestorOpInBlock(*user));
            });
          });
          if (!shared)
            continue;
          OpBuilder builder(&op);
          IRMapping mapping;
          for (Operation *view : llvm::reverse(chain)) {
            placed.insert(builder.clone(*view, mapping));
            sharedViews.insert(view);
          }
          op.setOperand(index, mapping.lookup(op.getOperand(index)));
        }
      }
      for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body)))
        if (sharedViews.contains(&op)) {
          placed.erase(&op);
          if (op.use_empty())
            op.erase();
        }

      // The subgraphs: placed operations joined by their values, each in
      // the order of the block, with the values of the CPU they read
      // through constants, empty tensors and fills. A subgraph is launched
      // as early as it can be, right after the last of those values (or at
      // the start of the block); the subgraphs in the order of their
      // launches, then of their first operations.
      llvm::EquivalenceClasses<Operation *> joined;
      for (Operation &op : body) {
        if (!placed.contains(&op))
          continue;
        joined.insert(&op);
        for (Value value : op.getOperands())
          if (Operation *def = value.getDefiningOp(); placed.contains(def)) {
            if (isRowOperator(&op) || isRowOperator(def) ||
                def->hasAttr("hart.row_producer"))
              continue;
            // A CPU consumer makes this result a task boundary. Joining past
            // it can make a later task depend on its own not-yet-launched
            // result.
            if (llvm::all_of(def->getUsers(), [&](Operation *user) {
                  return placed.contains(body.findAncestorOpInBlock(*user));
                }))
              joined.unionSets(&op, def);
          }
      }
      llvm::MapVector<Operation *, SmallVector<Operation *>> members;
      for (Operation &op : body)
        if (placed.contains(&op))
          members[joined.getLeaderValue(&op)].push_back(&op);
      struct Subgraph {
        Operation *latest;
        SmallVector<Operation *> ops;
        SetVector<Value> reads;
        SetVector<Operation *> locals;
      };
      SmallVector<Subgraph> subgraphs;
      for (auto &[leader, ops] : members) {
        Subgraph subgraph{nullptr, ops, {}, {}};
        DenseSet<Operation *> inside(ops.begin(), ops.end());
        std::function<void(Value)> visit = [&](Value value) {
          Operation *def = value.getDefiningOp();
          if (def && inside.contains(def))
            return;
          if (def && isLocal(def)) {
            if (subgraph.locals.insert(def))
              for (Value operand : def->getOperands())
                visit(operand);
            return;
          }
          subgraph.reads.insert(value);
        };
        for (Operation *op : ops)
          for (Value value : readsOf(op))
            visit(value);
        Operation *latest = nullptr;
        for (Value value : subgraph.reads) {
          Operation *def = value.getDefiningOp();
          if (!def)
            continue;
          def = body.findAncestorOpInBlock(*def);
          if (!latest || latest->isBeforeInBlock(def))
            latest = def;
        }
        subgraph.latest = latest;
        subgraphs.push_back(std::move(subgraph));
      }
      llvm::stable_sort(subgraphs, [](const Subgraph &a, const Subgraph &b) {
        if (a.latest == b.latest)
          return a.ops.front()->isBeforeInBlock(b.ops.front());
        return !a.latest || (b.latest && a.latest->isBeforeInBlock(b.latest));
      });

      // The transfer of each value of the CPU to the thread.
      DenseMap<std::pair<Value, int64_t>, Value> sent;
      // The constants, empty tensors and fills subgraphs read.
      SetVector<Operation *> locals;
      // The last launch after each last value of the CPU: the launches
      // after the same value follow each other.
      DenseMap<Operation *, Operation *> after;
      // The subgraphs follow their function, in order.
      Operation *previous = function;
      for (auto [number, subgraph] : llvm::enumerate(subgraphs)) {
        ArrayRef<Operation *> ops = subgraph.ops;
        Operation *&cursor = after[subgraph.latest];
        if (!cursor)
          cursor = subgraph.latest;
        // The launch goes before `point`.
        Operation *point = cursor ? cursor->getNextNode() : &body.front();
        int64_t thread = threads[number % threads.size()];
        IntegerAttr threadAttr =
            IntegerAttr::get(IntegerType::get(context, 64), thread);
        locals.insert(subgraph.locals.begin(), subgraph.locals.end());
        DenseSet<Operation *> inside(ops.begin(), ops.end());
        auto isInside = [&](Operation *user) {
          return inside.contains(body.findAncestorOpInBlock(*user));
        };
        SetVector<Value> results;
        for (Operation *op : ops)
          for (Value value : op->getResults())
            for (Operation *user : value.getUsers()) {
              if (isInside(user))
                continue;
              if (body.findAncestorOpInBlock(*user)->isBeforeInBlock(point)) {
                user->emitError("reads a value of a thread subgraph before "
                                "the subgraph is launched");
                return signalPassFailure();
              }
              results.insert(value);
            }

        // A thread reads and writes tensors.
        for (Value value : llvm::concat<const Value>(subgraph.reads, results))
          if (!isa<RankedTensorType>(value.getType())) {
            ops.front()->emitError("a thread subgraph reads or writes a value "
                                   "that is not a tensor");
            return signalPassFailure();
          }

        // The subgraph.
        Location loc = ops.front()->getLoc();
        OpBuilder builder(context);
        builder.setInsertionPointAfter(previous);
        auto outlined = func::FuncOp::create(
            builder, loc,
            (function.getName() + "_ht" + llvm::Twine(number)).str(),
            FunctionType::get(
                context,
                llvm::to_vector(llvm::concat<const Type>(
                    ValueRange(subgraph.reads.getArrayRef()).getTypes(),
                    ValueRange(results.getArrayRef()).getTypes())),
                ValueRange(results.getArrayRef()).getTypes()));
        outlined.setPrivate();
        for (Operation *op : ops)
          if (auto group = op->getAttr("hart.parallel_group")) {
            outlined->setAttr("hart.parallel_group", group);
            outlined->setAttr("hart.parallel_part",
                              op->getAttr("hart.parallel_part"));
          }
        if (llvm::any_of(ops, isRowOperator)) {
          for (auto [index, value] : llvm::enumerate(subgraph.reads)) {
            auto type = cast<RankedTensorType>(value.getType());
            SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
            strides.back() = 1;
            outlined.setArgAttr(
                index, "bufferization.buffer_layout",
                StridedLayoutAttr::get(context, ShapedType::kDynamic, strides));
            outlined.setArgAttr(index, "bufferization.writable",
                                builder.getBoolAttr(false));
          }
        }
        outlined->setAttr(
            "hart.program_bank",
            builder.getI64IntegerAttr(placement->virtual_bank_count()));
        outlined->setAttr(
            "bbacc.target",
            builder.getStringAttr(
                design.threads(placement->thread_indices(thread)).pkg()));
        previous = outlined;
        builder.setInsertionPointToStart(outlined.addEntryBlock());
        IRMapping mapping;
        mapping.map(subgraph.reads.getArrayRef(),
                    outlined.getArguments().take_front(subgraph.reads.size()));
        std::function<void(Value)> materialize = [&](Value value) {
          if (mapping.contains(value))
            return;
          Operation *def = value.getDefiningOp();
          for (Value operand : def->getOperands())
            materialize(operand);
          builder.clone(*def, mapping);
        };
        for (Operation *op : ops) {
          for (Value value : readsOf(op))
            materialize(value);
          builder.clone(*op, mapping);
        }
        // Each result written into its destination, an argument after the
        // inputs.
        SmallVector<Value> returned;
        for (auto [index, value] : llvm::enumerate(results))
          returned.push_back(
              bufferization::MaterializeInDestinationOp::create(
                  builder, loc, mapping.lookup(value),
                  outlined.getArgument(subgraph.reads.size() + index))
                  .getResult());
        func::ReturnOp::create(builder, loc, returned);

        // On the CPU: the transfers and the launch before `point`; each
        // result transferred back right before the first operation of the
        // CPU that reads it, so the CPU goes on while the thread runs.
        builder.setInsertionPoint(point);
        Sizes sizes{body, point, placed};
        auto empty = [&](Value value) {
          auto type = cast<RankedTensorType>(value.getType());
          return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                         type.getElementType(),
                                         sizes.dynamic(builder, loc, value))
              .getResult();
        };
        SmallVector<Value> operands;
        for (Value value : subgraph.reads) {
          Value &copy = sent[{value, thread}];
          if (!copy)
            copy = TransferOp::create(builder, loc, TypeRange{value.getType()},
                                      value, empty(value), IntegerAttr(),
                                      threadAttr)
                       .getResult()
                       .front();
          operands.push_back(copy);
        }
        SmallVector<Value> outs, backs;
        for (Value value : results) {
          outs.push_back(empty(value));
          backs.push_back(empty(value));
        }
        // The launch computes its results here: their locations.
        SmallVector<Location> computed;
        for (Value value : results)
          computed.push_back(value.getDefiningOp()->getLoc());
        auto launch =
            LaunchOp::create(builder, FusedLoc::get(context, computed),
                             ValueRange(results.getArrayRef()).getTypes(),
                             FlatSymbolRefAttr::get(outlined.getNameAttr()),
                             operands, outs, threadAttr);
        cursor = launch;
        for (auto [index, result] : llvm::enumerate(launch.getResults())) {
          Value value = results[index];
          Operation *first = nullptr;
          for (Operation *user : value.getUsers()) {
            if (isInside(user))
              continue;
            Operation *top = body.findAncestorOpInBlock(*user);
            if (!first || top->isBeforeInBlock(first))
              first = top;
          }
          builder.setInsertionPoint(first);
          Value back = TransferOp::create(
                           builder, loc, TypeRange{value.getType()}, result,
                           backs[index], threadAttr, IntegerAttr())
                           .getResult()
                           .front();
          value.replaceUsesWithIf(
              back, [&](OpOperand &use) { return !isInside(use.getOwner()); });
          // Later tasks also keep these values in their explicit input lists.
          for (Subgraph &pending : subgraphs)
            if (pending.reads.remove(value)) {
              pending.reads.insert(back);
              if (pending.latest == value.getDefiningOp() || !pending.latest ||
                  pending.latest->isBeforeInBlock(back.getDefiningOp()))
                pending.latest = back.getDefiningOp();
            }
        }
        for (Operation *op : llvm::reverse(ops))
          op->erase();
      }

      // The constants, empty tensors and fills only subgraphs read, users
      // before the values they read.
      SmallVector<Operation *> dead(locals.begin(), locals.end());
      llvm::sort(dead, [](Operation *a, Operation *b) {
        return b->isBeforeInBlock(a);
      });
      for (Operation *op : dead)
        if (op->use_empty())
          op->erase();
    }
  }
};

PassRegistration<ConvertLinalgToHartPass> registration;
} // namespace

void preparePolyViews(ModuleOp module) {
  SmallVector<linalg::TransposeOp> views;
  module.walk([&](linalg::TransposeOp op) {
    auto input = cast<RankedTensorType>(op.getInput().getType());
    if (!input.getElementType().isF32() || !decoding(op.getInput()).empty())
      return;
    int64_t previous = -1;
    for (int64_t axis : op.getPermutation()) {
      if (input.getDimSize(axis) == 1)
        continue;
      if (axis < previous)
        return;
      previous = axis;
    }
    views.push_back(op);
  });
  // Moving singleton axes preserves the element order and needs no copy.
  for (auto op : views) {
    OpBuilder b(op);
    auto loc = op.getLoc();
    auto input = cast<RankedTensorType>(op.getInput().getType());
    auto output = cast<RankedTensorType>(op->getResult(0).getType());
    ReassociationIndices axes;
    for (int64_t axis = 0; axis < input.getRank(); ++axis)
      axes.push_back(axis);
    auto flatType =
        RankedTensorType::get({input.hasStaticShape() ? input.getNumElements()
                                                      : ShapedType::kDynamic},
                              input.getElementType());
    Value flat = tensor::CollapseShapeOp::create(
        b, loc, flatType, op.getInput(), ArrayRef<ReassociationIndices>{axes});
    SmallVector<OpFoldResult> sizes;
    for (auto [axis, size] : llvm::enumerate(output.getShape()))
      sizes.push_back(
          size == ShapedType::kDynamic
              ? OpFoldResult(tensor::DimOp::create(b, loc, op.getInput(),
                                                   op.getPermutation()[axis])
                                 .getResult())
              : OpFoldResult(b.getIndexAttr(size)));
    axes.clear();
    for (int64_t axis = 0; axis < output.getRank(); ++axis)
      axes.push_back(axis);
    Value view = tensor::ExpandShapeOp::create(
        b, loc, output, flat, ArrayRef<ReassociationIndices>{axes}, sizes);
    op->getResult(0).replaceAllUsesWith(view);
    op.erase();
  }
  for (auto function : module.getOps<func::FuncOp>())
    if (!function.isExternal())
      foldZeroAddViews(function.getBody().front());
}
