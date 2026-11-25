//===- ConvertLinalgToHart.cpp - A graph on the poly threads --------------===//
//
// Place MXINT8 products, quantizations and outlined row operators on the
// hardware threads described by design.pb. Decoding and layout views stay
// with their products, so the Balls read packed data directly.
//
// Each subgraph selects threads from its own requirements. A wide GP operator
// does not exclude narrower threads from independent matrix or row work.
// Row partitions and connected pipelines use the threads implementing their
// operations, preserving fusion within that eligible pool.
//
// Launches follow their inputs and keep independent work asynchronous. Values
// crossing thread boundaries have explicit logical transfers, lowered by the
// hart compiler to kernel dependencies and transfer bindings.
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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
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
bool partitionRowPipeline(func::FuncOp function, ArrayRef<int64_t> threads,
                          ArrayRef<StringRef> profiles, int64_t programBank,
                          int64_t bankRows, int64_t linkBanks);

namespace {
bool isQuantize(Operation *op) {
  return op->getName().getStringRef() == "frontend.fp32_to_mxint8";
}

bool isDecode(Operation *op) {
  return op->getName().getStringRef() == "frontend.mxint8_to_fp32";
}

bool isMxint5Decode(Operation *op) {
  return op->getName().getStringRef() == "frontend.mxint5_to_fp32";
}

bool isPackedPlane(Operation *op) { return op->hasAttr("hart.packed_plane"); }

bool isActivation(Operation *op) {
  return op->getName().getStringRef() == "frontend.mxint8_activation";
}

bool isIntegerSoftmax(Operation *op) {
  return op->getName().getStringRef() == "frontend.integer_softmax";
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
    if (isDecode(op) || isMxint5Decode(op)) {
      // A weight padded to the Ball layout is decoded through its slices.
      for (Value source = op->getOperand(0);
           auto slice = source.getDefiningOp<tensor::ExtractSliceOp>();
           source = slice.getSource()) {
        if (!slice.hasZeroOffset() || slice->hasAttr("hart.columns"))
          break;
        chain.push_back(slice);
      }
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
    if (isActivation(def) || isIntegerSoftmax(def))
      return size(builder, loc, def->getOperand(0), dim);
    if (isQuantize(def) || isDecode(def) || isMxint5Decode(def)) {
      Value source = size(builder, loc, def->getOperand(0), dim);
      if (dim != cast<IntegerAttr>(def->getAttr("axis")).getInt())
        return source;
      // Along the MXINT8 axis: K values zero-padded to Kp = ceil(K / 32) * 32
      // pack into Kp + Kp / 32 bytes, which decode to the Kp values.
      Value width = arith::ConstantIndexOp::create(
          builder, loc, isMxint5Decode(def) ? 16 : 32);
      if (isDecode(def) || isMxint5Decode(def))
        return arith::MulIOp::create(
            builder, loc,
            arith::DivUIOp::create(
                builder, loc, source,
                arith::ConstantIndexOp::create(builder, loc,
                                               isMxint5Decode(def) ? 11 : 33)),
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

// Split packed projections by output channels. This uses every local thread
// for decode and avoids rereading all weights in each prefill row partition.
void partitionColumns(func::FuncOp function,
                      function_ref<int64_t(Operation *)> threadCount) {
  int64_t group = 0;
  for (auto matrix :
       llvm::to_vector(function.getBody().front().getOps<linalg::MatmulOp>())) {
    int64_t count = threadCount(matrix);
    if (count < 2)
      continue;
    auto type = cast<RankedTensorType>(matrix.getResult(0).getType());
    if (type.getRank() != 2 || type.isDynamicDim(1) || type.getDimSize(1) < 64)
      continue;
    auto maps = matrix.getIndexingMapsArray();
    auto column = dyn_cast<AffineDimExpr>(maps[1].getResult(0));
    if (!column || column.getPosition() != 1)
      continue;
    auto lhs = matrix.getInputs()[0].getDefiningOp();
    auto collapse =
        matrix.getInputs()[1].getDefiningOp<tensor::CollapseShapeOp>();
    auto transpose =
        collapse ? collapse.getSrc().getDefiningOp<linalg::TransposeOp>()
                 : linalg::TransposeOp();
    Operation *rhs = transpose ? transpose.getInput().getDefiningOp() : nullptr;
    auto fill = matrix.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    auto shape = fill ? fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()
                      : tensor::EmptyOp();
    if (!lhs || !rhs || !isDecode(lhs) ||
        (!isDecode(rhs) && !isMxint5Decode(rhs)) || !fill || !shape ||
        !matchPattern(fill.getInputs()[0], m_PosZeroFloat()) ||
        transpose.getPermutation() != ArrayRef<int64_t>{1, 0, 2} ||
        (collapse.getReassociationIndices() !=
             SmallVector<ReassociationIndices>{{0}, {1, 2}} &&
         collapse.getReassociationIndices() !=
             SmallVector<ReassociationIndices>{{0, 1}, {2}}))
      continue;
    auto packed = cast<RankedTensorType>(rhs->getOperand(0).getType());
    auto decoded = cast<RankedTensorType>(rhs->getResult(0).getType());
    if (!packed.hasStaticShape() || packed.getRank() != 3 ||
        packed.getDimSize(2) % 16 ||
        rhs->getOperand(0).getDefiningOp<tensor::ExtractSliceOp>() ||
        decoded.getDimSize(1) != type.getDimSize(1) ||
        rhs->getAttrOfType<IntegerAttr>("axis").getInt() != 2)
      continue;
    OpBuilder b(matrix);
    Location loc = matrix.getLoc();
    int64_t n = type.getDimSize(1), chunks = packed.getDimSize(0);
    int64_t k = decoded.getDimSize(2), width = packed.getDimSize(2);
    if (!isMxint5Decode(rhs) && n < count * 32)
      continue;
    Operation *quantized = nullptr;
    if (isMxint5Decode(rhs)) {
      if (n % 32 || !matrix.getResult(0).hasOneUse())
        continue;
      Operation *consumer = *matrix.getResult(0).getUsers().begin();
      if (!isQuantize(consumer) ||
          consumer->getAttrOfType<IntegerAttr>("axis").getInt() != 1)
        continue;
      quantized = consumer;
    }
    int64_t parts = quantized ? std::min(count, n / 32) : count;
    Value height =
        type.isDynamicDim(0)
            ? shape.getDynamicSizes().front()
            : arith::ConstantIndexOp::create(b, loc, type.getDimSize(0));
    OpFoldResult rowSize =
        type.isDynamicDim(0) ? OpFoldResult(height)
                             : OpFoldResult(b.getIndexAttr(type.getDimSize(0)));
    SmallVector<Value> dynamic;
    if (type.isDynamicDim(0))
      dynamic.push_back(height);
    auto resultType =
        quantized ? cast<RankedTensorType>(quantized->getResult(0).getType())
                  : type;
    // The assembled result is the quantize's value when the split carries it.
    Location resultLoc = quantized ? quantized->getLoc() : loc;
    Value result =
        tensor::EmptyOp::create(b, resultLoc, resultType.getShape(),
                                resultType.getElementType(), dynamic);
    StringAttr identity = b.getStringAttr(
        (function.getName() + "_columns" + llvm::Twine(group++)).str());
    int64_t tiles = (n + 31) / 32;
    for (int64_t part = 0; part < parts; ++part) {
      int64_t start = std::min(n, tiles * part / parts * 32);
      int64_t length = std::min(n, tiles * (part + 1) / parts * 32) - start;
      Value slice = tensor::ExtractSliceOp::create(
          b, loc, rhs->getOperand(0),
          ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(start),
                                 b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(chunks), b.getIndexAttr(length),
                                 b.getIndexAttr(width)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                 b.getIndexAttr(1)});
      slice.getDefiningOp()->setAttr("hart.columns", b.getI64IntegerAttr(n));
      Operation *a = b.clone(*lhs), *w = b.clone(*rhs);
      w->setOperand(0, slice);
      w->getResult(0).setType(
          RankedTensorType::get({chunks, length, k}, type.getElementType()));
      Value transposed = tensor::EmptyOp::create(
          b, loc, ArrayRef<int64_t>{length, chunks, k}, type.getElementType());
      transposed =
          linalg::TransposeOp::create(b, loc, w->getResult(0), transposed,
                                      ArrayRef<int64_t>{1, 0, 2})
              ->getResult(0);
      Value decodedWeights = tensor::CollapseShapeOp::create(
          b, loc,
          RankedTensorType::get({length, chunks * k}, type.getElementType()),
          transposed, ArrayRef<ReassociationIndices>{{0}, {1, 2}});
      Value empty = tensor::EmptyOp::create(
          b, loc, ArrayRef<int64_t>{type.getDimSize(0), length},
          type.getElementType(), dynamic);
      Value init =
          linalg::FillOp::create(b, loc, fill.getInputs(), ValueRange{empty})
              .getResult(0);
      IRMapping mapping;
      mapping.map(matrix.getInputs()[0], a->getResult(0));
      mapping.map(matrix.getInputs()[1], decodedWeights);
      mapping.map(matrix.getOutputs()[0], init);
      Operation *task = b.clone(*matrix, mapping);
      task->getResult(0).setType(RankedTensorType::get(
          {type.getDimSize(0), length}, type.getElementType()));
      task->setAttr("hart.parallel_group", identity);
      task->setAttr("hart.parallel_part", b.getI64IntegerAttr(part));
      task->setAttr("hart.columns", b.getI64IntegerAttr(n));
      if (quantized) {
        IRMapping quantMapping;
        quantMapping.map(matrix.getResult(0), task->getResult(0));
        Operation *q = b.clone(*quantized, quantMapping);
        q->getResult(0).setType(
            RankedTensorType::get({type.getDimSize(0), length + length / 32},
                                  resultType.getElementType()));
        for (int64_t plane = 0; plane < 2; ++plane) {
          int64_t offset = plane ? length : 0;
          int64_t columns = plane ? length / 32 : length;
          int64_t destination = plane ? n + start / 32 : start;
          Value view = tensor::ExtractSliceOp::create(
              b, resultLoc, q->getResult(0),
              ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(offset)},
              ArrayRef<OpFoldResult>{rowSize, b.getIndexAttr(columns)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          view.getDefiningOp()->setAttr("hart.packed_plane", b.getUnitAttr());
          result = tensor::InsertSliceOp::create(
              b, resultLoc, view, result,
              ArrayRef<OpFoldResult>{b.getIndexAttr(0),
                                     b.getIndexAttr(destination)},
              ArrayRef<OpFoldResult>{rowSize, b.getIndexAttr(columns)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        }
      } else {
        result = tensor::InsertSliceOp::create(
            b, loc, task->getResult(0), result,
            ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(start)},
            ArrayRef<OpFoldResult>{rowSize, b.getIndexAttr(length)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
      }
    }
    if (quantized) {
      quantized->getResult(0).replaceAllUsesWith(result);
      quantized->erase();
    } else {
      matrix.getResult(0).replaceAllUsesWith(result);
    }
    matrix.erase();
  }
}

// Split contiguous outer row ranges across the local thread pool.
void partitionRows(func::FuncOp function,
                   function_ref<int64_t(Operation *)> threadCount,
                   bool batched) {
  Block &body = function.getBody().front();
  DenseSet<Operation *> shapeOps;
  for (Operation &op : body)
    if (isDecode(&op) || isQuantize(&op) || isActivation(&op) ||
        isIntegerSoftmax(&op) || isRowOperator(&op) ||
        isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
      shapeOps.insert(&op);
  int64_t group = 0;
  for (Operation *op : llvm::to_vector(
           llvm::map_range(body, [](Operation &op) { return &op; }))) {
    auto matrix = dyn_cast<linalg::MatmulOp>(op);
    if (matrix && op->hasAttr("hart.parallel_group"))
      continue;
    if (matrix && llvm::any_of(decoding(matrix.getInputs()[1]), isMxint5Decode))
      continue;
    Operation *decode =
        matrix ? matrix.getInputs()[0].getDefiningOp() : nullptr;
    if (matrix && (!decode || !isDecode(decode) ||
                   !isa<BlockArgument>(decode->getOperand(0))))
      continue;
    auto fill = matrix ? matrix.getOutputs()[0].getDefiningOp<linalg::FillOp>()
                       : linalg::FillOp();
    if (matrix &&
        !matchPattern(fill ? fill.getInputs()[0] : matrix.getOutputs()[0],
                      m_AnyZeroFloat()))
      continue;
    if (!isRowOperator(op) && !isActivation(op) && !isIntegerSoftmax(op) &&
        !matrix)
      continue;
    int64_t count = threadCount(op);
    if (count < 2)
      continue;
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    if (type.getRank() < 2)
      continue;
    if (matrix && batched && (type.isDynamicDim(0) || type.getDimSize(0) <= 1))
      continue;
    int64_t axis = 0;
    while (axis < type.getRank() - 2 && !type.isDynamicDim(axis) &&
           type.getDimSize(axis) < count)
      ++axis;
    if (!type.isDynamicDim(axis) && type.getDimSize(axis) < count)
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
    SmallVector<Value> bounds{index(0)};
    for (int64_t part = 1; part < count; ++part)
      bounds.push_back(arith::DivUIOp::create(
          b, loc, arith::MulIOp::create(b, loc, rows, index(part)),
          index(count)));
    bounds.push_back(rows);
    SmallVector<Value> parts, lengths;
    StringAttr identity = b.getStringAttr(
        (function.getName() + "_rows" + llvm::Twine(group++)).str());
    for (int64_t part = 0; part < count; ++part) {
      int64_t staticLength = type.isDynamicDim(axis)
                                 ? ShapedType::kDynamic
                                 : type.getDimSize(axis) * (part + 1) / count -
                                       type.getDimSize(axis) * part / count;
      Value length = ShapedType::isDynamic(staticLength)
                         ? Value(arith::SubIOp::create(b, loc, bounds[part + 1],
                                                       bounds[part]))
                         : index(staticLength);
      lengths.push_back(length);
      IRMapping mapping;
      auto maps = matrix
                      ? b.getAffineMapArrayAttr(matrix.getIndexingMapsArray())
                      : op->getAttrOfType<ArrayAttr>("maps");
      if (isActivation(op)) {
        auto identity = b.getMultiDimIdentityMap(type.getRank());
        auto table =
            AffineMap::get(type.getRank(), 0, b.getAffineConstantExpr(0));
        maps = b.getAffineMapArrayAttr({identity, identity, table, identity});
      }
      if (isIntegerSoftmax(op)) {
        auto identity = b.getMultiDimIdentityMap(type.getRank());
        auto table =
            AffineMap::get(type.getRank(), 0, b.getAffineConstantExpr(0));
        maps = b.getAffineMapArrayAttr({identity, table, identity});
      }
      for (auto [input, attr] :
           llvm::zip_equal(op->getOperands().take_front(maps.size() - 1),
                           maps.getValue().drop_back())) {
        auto map = cast<AffineMapAttr>(attr).getValue();
        for (auto [dim, expression] : llvm::enumerate(map.getResults())) {
          auto d = dyn_cast<AffineDimExpr>(expression);
          if (!d || d.getPosition() != axis)
            continue;
          Value data = matrix ? decode->getOperand(0) : input;
          auto source = cast<RankedTensorType>(data.getType());
          SmallVector<OpFoldResult> offsets(source.getRank(),
                                            b.getIndexAttr(0));
          SmallVector<OpFoldResult> sizes,
              strides(source.getRank(), b.getIndexAttr(1));
          for (int64_t i = 0; i < source.getRank(); ++i)
            sizes.push_back(
                i == int64_t(dim)
                    ? (ShapedType::isDynamic(staticLength)
                           ? OpFoldResult(length)
                           : OpFoldResult(b.getIndexAttr(staticLength)))
                : source.isDynamicDim(i)
                    ? OpFoldResult(extent(data, i))
                    : OpFoldResult(b.getIndexAttr(source.getDimSize(i))));
          offsets[dim] = bounds[part];
          Value slice = tensor::ExtractSliceOp::create(b, loc, data, offsets,
                                                       sizes, strides);
          if (matrix) {
            Operation *decoded = b.clone(*decode);
            decoded->setOperand(0, slice);
            auto shape = llvm::to_vector(
                cast<RankedTensorType>(input.getType()).getShape());
            shape[axis] = staticLength;
            decoded->getResult(0).setType(
                RankedTensorType::get(shape, type.getElementType()));
            slice = decoded->getResult(0);
          }
          mapping.map(input, slice);
        }
      }
      if (matrix) {
        SmallVector<int64_t> shape(type.getShape());
        shape[axis] = staticLength;
        SmallVector<Value> dynamic;
        for (int64_t i = 0; i < type.getRank(); ++i)
          if (type.isDynamicDim(i))
            dynamic.push_back(i == axis ? length : extent(op->getResult(0), i));
        Value empty = tensor::EmptyOp::create(b, loc, shape,
                                              type.getElementType(), dynamic);
        Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0));
        mapping.map(
            matrix.getOutputs()[0],
            linalg::FillOp::create(b, loc, ValueRange{zero}, ValueRange{empty})
                .getResult(0));
      }
      Operation *task = b.clone(*op, mapping);
      SmallVector<int64_t> shape(type.getShape());
      shape[axis] = staticLength;
      task->getResult(0).setType(
          RankedTensorType::get(shape, type.getElementType()));
      task->setAttr("hart.parallel_group", identity);
      task->setAttr("hart.parallel_part", b.getI64IntegerAttr(part));
      parts.push_back(task->getResult(0));
    }
    Value result;
    if (op->getResult(0).hasOneUse())
      if (auto destination =
              dyn_cast<bufferization::MaterializeInDestinationOp>(
                  *op->getResult(0).getUsers().begin());
          destination && destination.getDest().getType() == type)
        if (auto argument = dyn_cast<BlockArgument>(destination.getDest());
            argument && argument.getOwner() == &body) {
          result = argument;
          function.setArgAttr(argument.getArgNumber(), "bufferization.writable",
                              b.getBoolAttr(true));
        }
    if (!result)
      result = tensor::EmptyOp::create(
          b, loc, type.getShape(), type.getElementType(),
          shapes.dynamic(b, loc, op->getResult(0)));
    for (int64_t part = 0; part < count; ++part) {
      SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes,
          strides(type.getRank(), b.getIndexAttr(1));
      offsets[axis] = bounds[part];
      auto partType = cast<RankedTensorType>(parts[part].getType());
      for (int64_t dim = 0; dim < type.getRank(); ++dim)
        sizes.push_back(
            dim == axis
                ? (partType.isDynamicDim(dim)
                       ? OpFoldResult(lengths[part])
                       : OpFoldResult(b.getIndexAttr(partType.getDimSize(dim))))
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
  memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanup);
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
    if (failed(outlineOperators(getOperation())))
      return signalPassFailure();
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
    // The hart knows each thread's actual Balls and GP domain. A subgraph
    // uses only the threads that implement its computations.
    SmallVector<int64_t> computeThreads;
    SmallVector<StringRef> profiles;
    for (auto [local, index] : llvm::enumerate(placement->thread_indices())) {
      const auto &thread = design.threads(index);
      computeThreads.push_back(local);
      profiles.push_back(thread.pkg());
    }

    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    // Resolve shape-only uses before selecting computations for threads.
    RewritePatternSet shapes(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(shapes);
    if (failed(applyPatternsGreedily(module, std::move(shapes))))
      return signalPassFailure();
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal() || function->hasAttr("bbacc.gp_operator") ||
          function->hasAttr("bbacc.target"))
        continue;
      // Keep the host argument ABI explicit while inferring borrowed results.
      for (auto argument : function.getArguments())
        if (auto type = dyn_cast<RankedTensorType>(argument.getType());
            type && !function.getArgAttr(argument.getArgNumber(),
                                         "bufferization.buffer_layout"))
          function.setArgAttr(
              argument.getArgNumber(), "bufferization.buffer_layout",
              AffineMapAttr::get(
                  AffineMap::getMultiDimIdentityMap(type.getRank(), context)));
      Block &body = function.getBody().front();
      foldZeroAddViews(body);
      // Match the requirements of each subgraph, rather than excluding a
      // thread from every stage because another stage needs a wider GP type.
      auto eligible = [&](ArrayRef<Operation *> operations) {
        bool needsGP = false, needsWideGP = false;
        bool needsMatrix = false, needsMatrix5 = false;
        bool needsQuant = false, needsActivation = false;
        bool needsSoftmax = false;
        for (Operation *op : operations) {
          if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
            bool mx5 =
                llvm::any_of(decoding(op->getOperand(1)), isMxint5Decode);
            needsMatrix |= !mx5;
            needsMatrix5 |= mx5;
          }
          needsActivation |= isActivation(op);
          needsSoftmax |= isIntegerSoftmax(op);
          needsQuant |= isQuantize(op);
          if (!isRowOperator(op))
            continue;
          needsGP = true;
          auto kernel = module.lookupSymbol<func::FuncOp>(
              op->getAttrOfType<FlatSymbolRefAttr>("operator").getValue());
          kernel.walk([&](Operation *nested) {
            needsWideGP |=
                llvm::any_of(nested->getResultTypes(), [](Type type) {
                  return getElementTypeOrSelf(type).isF64();
                });
          });
        }
        SmallVector<int64_t> candidates;
        for (int64_t thread : computeThreads) {
          const auto &instance =
              design.threads(placement->thread_indices(thread));
          const auto &gp = instance.gpdomain();
          llvm::StringSet<> balls;
          for (const auto &ball : instance.balldomain().balls())
            balls.insert(ball.component());
          if ((!needsMatrix || balls.contains("mxint8matmul")) &&
              (!needsMatrix5 || balls.contains("mxint5gemv")) &&
              (!needsQuant || balls.contains("mxint8quantize")) &&
              (!needsActivation || balls.contains("mxint8act")) &&
              (!needsSoftmax || balls.contains("intsoftmax")) &&
              (!needsGP || gp.enable()) && (!needsWideGP || gp.e_len() >= 64))
            candidates.push_back(thread);
        }
        return candidates;
      };
      SmallVector<Operation *> operations;
      for (Operation &op : body)
        operations.push_back(&op);
      SmallVector<int64_t> available = eligible(operations);
      SmallVector<StringRef> availableProfiles;
      for (auto [i, thread] : llvm::enumerate(computeThreads)) {
        if (llvm::is_contained(available, thread))
          availableProfiles.push_back(profiles[i]);
      }
      bool batched = llvm::any_of(
          body, [](Operation &op) { return isa<linalg::BatchMatmulOp>(op); });
      int64_t linkBanks = 0;
      for (const auto &link : placement->data_links())
        if (llvm::all_of(available, [&](int64_t thread) {
              return llvm::is_contained(link.threads(), thread);
            }))
          linkBanks = std::max<int64_t>(linkBanks, link.banks());
      if (!batched && !available.empty() &&
          partitionRowPipeline(
              function, available, availableProfiles,
              placement->virtual_bank_count(),
              design.threads(placement->thread_indices(available.front()))
                  .mem()
                  .bank()
                  .entries(),
              linkBanks))
        continue;
      // Independent Q/K/V projections retain their weights per thread.
      // Pointwise attention stages still use the full row pool.
      partitionColumns(function, [&](Operation *op) {
        return eligible(ArrayRef<Operation *>{op}).size();
      });
      partitionRows(
          function,
          [&](Operation *op) {
            return eligible(ArrayRef<Operation *>{op}).size();
          },
          batched);
      DenseSet<Operation *> placed;
      for (Operation &op : body) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        placed.insert(&op);
        SmallVector<Operation *> lhs = decoding(op.getOperand(0)),
                                 rhs = decoding(op.getOperand(1));
        if (lhs.empty() || rhs.empty()) {
          op.emitError("poly matrix products require packed operands");
          return signalPassFailure();
        }
        placed.insert(lhs.begin(), lhs.end());
        placed.insert(rhs.begin(), rhs.end());
      }
      for (Operation &op : body)
        if ((isQuantize(&op) || isActivation(&op) || isIntegerSoftmax(&op) ||
             isRowOperator(&op) || isPackedPlane(&op)) &&
            !op.use_empty())
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

      // Keep shared decoding/layout views private to each product. Sharing a
      // packed input must not join independent products into one thread task.
      DenseSet<Operation *> sharedViews;
      for (Operation &op : llvm::make_early_inc_range(body)) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        for (unsigned index = 0; index < 2; ++index) {
          auto chain = decoding(op.getOperand(index));
          bool shared = llvm::any_of(chain, [&](Operation *view) {
            if (sharedViews.contains(view))
              return true;
            return llvm::any_of(view->getUsers(), [&](Operation *user) {
              Operation *consumer = body.findAncestorOpInBlock(*user);
              return consumer != &op && !llvm::is_contained(chain, consumer);
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
            if (isRowOperator(&op) || isRowOperator(def))
              continue;
            // A shared producer is a task boundary. Fusing through it would
            // merge independent column consumers back onto one thread.
            if (isPackedPlane(&op) ||
                llvm::all_of(def->getUsers(),
                             [&](Operation *user) { return user == &op; }))
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
        SmallVector<int64_t> threads = eligible(ops);
        if (threads.empty()) {
          ops.front()->emitError(
              "no thread implements this subgraph's Balls and GP types");
          return signalPassFailure();
        }
        // Independent groups can interleave; each group's parts retain their
        // distinct slots within the eligible thread pool.
        int64_t part = number % threads.size();
        for (Operation *op : ops)
          if (auto assigned =
                  op->getAttrOfType<IntegerAttr>("hart.parallel_part"))
            part = assigned.getInt();
        int64_t thread = threads[part];
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
        size_t symbolNumber = number;
        std::string name =
            (function.getName() + "_ht" + llvm::Twine(symbolNumber)).str();
        while (module.lookupSymbol(name))
          name =
              (function.getName() + "_ht" + llvm::Twine(++symbolNumber)).str();
        auto outlined = func::FuncOp::create(
            builder, loc, name,
            FunctionType::get(
                context,
                llvm::to_vector(llvm::concat<const Type>(
                    ValueRange(subgraph.reads.getArrayRef()).getTypes(),
                    ValueRange(results.getArrayRef()).getTypes())),
                ValueRange(results.getArrayRef()).getTypes()));
        outlined.setPrivate();
        for (auto [index, argumentType] :
             llvm::enumerate(outlined.getArgumentTypes())) {
          auto type = cast<RankedTensorType>(argumentType);
          outlined.setArgAttr(index, "bufferization.buffer_layout",
                              AffineMapAttr::get(builder.getMultiDimIdentityMap(
                                  type.getRank())));
        }
        for (auto [index, value] : llvm::enumerate(results))
          outlined.setArgAttr(subgraph.reads.size() + index,
                              "bufferization.writable",
                              builder.getBoolAttr(true));
        for (Operation *op : ops)
          if (auto group = op->getAttr("hart.parallel_group")) {
            outlined->setAttr("hart.parallel_group", group);
            outlined->setAttr("hart.parallel_part",
                              op->getAttr("hart.parallel_part"));
          }
        bool columns = llvm::any_of(
            ops, [](Operation *op) { return op->hasAttr("hart.columns"); });
        int64_t columnStride = 0;
        for (Operation *op : ops)
          if (auto width = op->getAttrOfType<IntegerAttr>("hart.columns"))
            columnStride = width.getInt();
        if (llvm::any_of(ops,
                         [](Operation *op) {
                           return isRowOperator(op) || isIntegerSoftmax(op);
                         }) ||
            columns) {
          for (auto [index, value] : llvm::enumerate(subgraph.reads)) {
            auto type = cast<RankedTensorType>(value.getType());
            if (!type.getRank())
              continue;
            SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
            strides.back() = 1;
            if (columns)
              for (int64_t axis = type.getRank() - 2; axis >= 0; --axis)
                strides[axis] = strides[axis + 1] * type.getDimSize(axis + 1);
            if (columns && type.getRank() == 3)
              strides[0] = columnStride * type.getDimSize(2);
            outlined.setArgAttr(
                index, "bufferization.buffer_layout",
                StridedLayoutAttr::get(context, ShapedType::kDynamic, strides));
            outlined.setArgAttr(index, "bufferization.writable",
                                builder.getBoolAttr(false));
          }
          for (auto [index, value] : llvm::enumerate(results)) {
            auto type = cast<RankedTensorType>(value.getType());
            SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
            strides.back() = 1;
            if (columns)
              strides[0] = columnStride + (isPackedPlane(value.getDefiningOp())
                                               ? columnStride / 32
                                               : 0);
            outlined.setArgAttr(
                subgraph.reads.size() + index, "bufferization.buffer_layout",
                StridedLayoutAttr::get(context, ShapedType::kDynamic, strides));
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

// A broadcast of decoded MXINT8 values that copies them (or adds +0) along
// axes other than their blocks, such as one key head shared by the query
// heads of a group, moves onto the packed values: the copy is of bytes, and
// the product reads one decoding of the broadcast, as the Ball reads packed
// operands. The blocked axis must stay one axis through the reshapes between
// the decoding and the broadcast.
void hoistDecodedBroadcasts(ModuleOp module) {
  SmallVector<linalg::GenericOp> broadcasts;
  module.walk([&](linalg::GenericOp op) {
    if (op.getNumDpsInputs() != 1 || op.getNumResults() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops() ||
        !op.getIndexingMapsArray().back().isIdentity())
      return;
    Block &body = op.getRegion().front();
    Value yielded = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
    if (yielded != body.getArgument(0)) {
      auto add = yielded.getDefiningOp<arith::AddFOp>();
      if (!add || std::distance(body.begin(), body.end()) != 2 ||
          !llvm::is_contained(add->getOperands(), body.getArgument(0)) ||
          !positiveZero(add.getLhs() == body.getArgument(0) ? add.getRhs()
                                                            : add.getLhs()))
        return;
    }
    broadcasts.push_back(op);
  });
  for (linalg::GenericOp op : broadcasts) {
    // The reshapes from the decoding to the broadcast, and the blocked axis
    // at the broadcast's input.
    SmallVector<Operation *> reshapes;
    Value value = op.getDpsInputs()[0];
    Operation *decode = nullptr;
    while (Operation *def = value.getDefiningOp()) {
      if (isDecode(def)) {
        decode = def;
        break;
      }
      if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(def))
        break;
      reshapes.push_back(def);
      value = def->getOperand(0);
    }
    if (!decode)
      continue;
    // The blocked axis through each reshape: its group may add or drop
    // unit axes around it, nothing else.
    auto through = [](ArrayRef<ReassociationIndices> groups,
                      RankedTensorType wide, bool expand,
                      int64_t axis) -> int64_t {
      for (auto [index, group] : llvm::enumerate(groups)) {
        if (expand ? (int64_t)index != axis : !llvm::is_contained(group, axis))
          continue;
        int64_t kept = -1;
        for (int64_t member : group)
          if (member != (expand ? -1 : axis) && wide.getDimSize(member) != 1) {
            if (!expand || kept >= 0)
              return -1;
            kept = member;
          }
        return expand ? (kept >= 0 ? kept : group.back()) : (int64_t)index;
      }
      return -1;
    };
    int64_t axis = cast<IntegerAttr>(decode->getAttr("axis")).getInt();
    for (Operation *reshape : llvm::reverse(reshapes)) {
      if (axis < 0)
        break;
      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(reshape))
        axis = through(expand.getReassociationIndices(), expand.getResultType(),
                       true, axis);
      else {
        auto collapse = cast<tensor::CollapseShapeOp>(reshape);
        axis = through(collapse.getReassociationIndices(),
                       collapse.getSrcType(), false, axis);
      }
    }
    if (axis < 0)
      continue;
    auto maps = op.getIndexingMapsArray();
    auto read = dyn_cast<AffineDimExpr>(maps[0].getResult(axis));
    if (!read)
      continue;
    int64_t outputAxis = read.getPosition();
    OpBuilder b(op);
    Location loc = op.getLoc();
    // The same reshapes of the packed values, the blocked axis `at` of
    // `packed` as long as its packed blocks.
    Value packed = decode->getOperand(0);
    int64_t at = cast<IntegerAttr>(decode->getAttr("axis")).getInt();
    auto packedType = [&](RankedTensorType type, int64_t axis) {
      SmallVector<int64_t> shape(type.getShape());
      shape[axis] = cast<RankedTensorType>(packed.getType()).getDimSize(at);
      return RankedTensorType::get(shape, IntegerType::get(op.getContext(), 8));
    };
    auto packedSize = [&]() -> OpFoldResult {
      int64_t size = cast<RankedTensorType>(packed.getType()).getDimSize(at);
      if (size != ShapedType::kDynamic)
        return b.getIndexAttr(size);
      return tensor::DimOp::create(b, loc, packed, at).getResult();
    };
    for (Operation *reshape : llvm::reverse(reshapes)) {
      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(reshape)) {
        auto groups = expand.getReassociationIndices();
        int64_t next = through(groups, expand.getResultType(), true, at);
        SmallVector<OpFoldResult> sizes = expand.getMixedOutputShape();
        sizes[next] = packedSize();
        auto type = packedType(expand.getResultType(), next);
        packed =
            tensor::ExpandShapeOp::create(b, loc, type, packed, groups, sizes);
        at = next;
      } else {
        auto collapse = cast<tensor::CollapseShapeOp>(reshape);
        auto groups = collapse.getReassociationIndices();
        int64_t next = through(groups, collapse.getSrcType(), false, at);
        auto type = packedType(collapse.getResultType(), next);
        packed = tensor::CollapseShapeOp::create(b, loc, type, packed, groups);
        at = next;
      }
    }
    auto result = cast<RankedTensorType>(op.getResult(0).getType());
    auto packedResult = packedType(result, outputAxis);
    SmallVector<Value> dynamic;
    for (int64_t dim = 0; dim < result.getRank(); ++dim)
      if (packedResult.isDynamicDim(dim))
        dynamic.push_back(
            dim == outputAxis
                ? tensor::DimOp::create(b, loc, packed, at).getResult()
                : tensor::DimOp::create(b, loc, op.getDpsInits()[0], dim)
                      .getResult());
    Value init = tensor::EmptyOp::create(b, loc, packedResult, dynamic);
    auto copy = linalg::GenericOp::create(
        b, loc, TypeRange{packedResult}, ValueRange{packed}, ValueRange{init},
        maps, op.getIteratorTypesArray(),
        [](OpBuilder &nested, Location nestedLoc, ValueRange arguments) {
          linalg::YieldOp::create(nested, nestedLoc, arguments[0]);
        });
    Operation *decoded = b.clone(*decode);
    decoded->setOperand(0, copy.getResult(0));
    decoded->getResult(0).setType(result);
    decoded->setAttr("axis", b.getI64IntegerAttr(outputAxis));
    op.getResult(0).replaceAllUsesWith(decoded->getResult(0));
    op.erase();
  }
}

void preparePolyViews(ModuleOp module) {
  // A rotate-half expression reads one source row; its concatenation need not
  // allocate a second vector and exceed the GP operator's three input groups.
  SmallVector<tensor::ConcatOp> concatenations;
  module.walk(
      [&](tensor::ConcatOp concat) { concatenations.push_back(concat); });
  llvm::for_each(concatenations, [&](tensor::ConcatOp concat) {
    auto type = cast<RankedTensorType>(concat.getResult().getType());
    int64_t rank = type.getRank();
    if (rank == 0 || concat.getDim() != uint64_t(rank - 1) ||
        !type.getElementType().isF32() || concat.getInputs().size() != 2 ||
        type.getShape().back() < 64 || type.getShape().back() % 64)
      return;
    auto negate = concat.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    auto low = concat.getInputs()[1].getDefiningOp<tensor::ExtractSliceOp>();
    if (!negate || !low || negate.getNumDpsInputs() != 1 ||
        negate.getNumParallelLoops() != rank ||
        negate.getRegion().front().getOperations().size() != 2 ||
        !llvm::all_of(negate.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }))
      return;
    auto negative = dyn_cast<arith::NegFOp>(negate.getRegion().front().front());
    auto high =
        negate.getDpsInputs()[0].getDefiningOp<tensor::ExtractSliceOp>();
    if (!negative ||
        negative.getOperand() != negate.getRegion().front().getArgument(0) ||
        negate.getRegion().front().getTerminator()->getOperand(0) !=
            negative.getResult() ||
        !high || high.getSource() != low.getSource() ||
        high.getMixedSizes() != low.getMixedSizes() || !low.hasUnitStride() ||
        !high.hasUnitStride() ||
        cast<RankedTensorType>(low.getSource().getType()) != type)
      return;
    int64_t half = type.getShape().back() / 2;
    for (int64_t axis = 0; axis < rank; ++axis)
      if (getConstantIntValue(low.getMixedOffsets()[axis]) != 0 ||
          getConstantIntValue(high.getMixedOffsets()[axis]) !=
              (axis == rank - 1 ? half : 0) ||
          (axis == rank - 1 &&
           getConstantIntValue(low.getMixedSizes()[axis]) != half))
        return;
    // The fused consumer already requires equal prefix extents for the source
    // and concatenation. Do not assume unrelated dynamic dimension SSA values
    // are equal when recognizing a standalone concatenation.
    linalg::GenericOp consumer;
    for (Operation *user : concat.getResult().getUsers()) {
      if (auto dim = dyn_cast<tensor::DimOp>(user)) {
        if (!dim.getConstantIndex())
          return;
        continue;
      }
      auto generic = dyn_cast<linalg::GenericOp>(user);
      if (!generic || (consumer && consumer != generic) ||
          !generic.hasPureTensorSemantics() || generic.getNumResults() != 1 ||
          generic.getNumLoops() != generic.getNumParallelLoops() ||
          generic.getNumLoops() != rank)
        return;
      consumer = generic;
    }
    if (!consumer)
      return;
    auto maps = consumer.getIndexingMapsArray();
    auto inputs = consumer.getDpsInputs();
    auto original = llvm::find(inputs, low.getSource());
    auto rotated = llvm::find(inputs, concat.getResult());
    if (original == inputs.end() || rotated == inputs.end() ||
        llvm::count(inputs, concat.getResult()) != 1 ||
        !maps[original - inputs.begin()].isIdentity() ||
        !maps[rotated - inputs.begin()].isIdentity())
      return;
    int64_t input = rotated - inputs.begin();
    OpBuilder b(consumer);
    auto loc = consumer.getLoc();
    SmallVector<AffineExpr> indices;
    for (int64_t axis = 0; axis < rank; ++axis)
      indices.push_back(b.getAffineDimExpr(axis));
    indices.back() = (indices.back() + half) % type.getShape().back();
    maps[input] = AffineMap::get(rank, 0, indices, module.getContext());
    consumer->setAttr("indexing_maps", b.getAffineMapArrayAttr(maps));
    consumer->setOperand(input, low.getSource());
    Value threshold = arith::ConstantIndexOp::create(b, loc, half);
    b.setInsertionPointToStart(&consumer.getRegion().front());
    Value paired = consumer.getRegion().front().getArgument(input);
    Value column = linalg::IndexOp::create(b, loc, rank - 1);
    Value lower = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                        column, threshold);
    Value minus = arith::NegFOp::create(b, loc, paired);
    Value signedValue = arith::SelectOp::create(b, loc, lower, minus, paired);
    for (OpOperand &use : llvm::make_early_inc_range(paired.getUses()))
      if (use.getOwner() != minus.getDefiningOp() &&
          use.getOwner() != signedValue.getDefiningOp())
        use.set(signedValue);
    SmallVector<tensor::DimOp> dimensions;
    for (Operation *user : concat.getResult().getUsers())
      dimensions.push_back(cast<tensor::DimOp>(user));
    for (auto dim : dimensions) {
      b.setInsertionPoint(dim);
      int64_t axis = *dim.getConstantIndex();
      Value size = axis == rank - 1
                       ? Value(arith::ConstantIndexOp::create(b, loc, half * 2))
                       : getValueOrCreateConstantIndexOp(
                             b, loc, low.getMixedSizes()[axis]);
      dim.replaceAllUsesWith(size);
      dim.erase();
    }
    concat.erase();
    if (negate->use_empty())
      negate.erase();
  });
  hoistDecodedBroadcasts(module);
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
  // A permutation of complete contiguous rows can use the GP row pipeline.
  // Keep decoded matrix layouts visible to their packed matrix consumers.
  SmallVector<linalg::TransposeOp> rows;
  module.walk([&](linalg::TransposeOp op) {
    auto input = cast<RankedTensorType>(op.getInput().getType());
    if (!op->getParentOfType<func::FuncOp>()->hasAttr("bbacc.target") &&
        input.getElementType().isF32() && input.getRank() > 1 &&
        input.getShape().back() >= 16 && input.getShape().back() % 4 == 0 &&
        op.getPermutation().back() == input.getRank() - 1 &&
        decoding(op.getInput()).empty())
      rows.push_back(op);
  });
  IRRewriter rewriter(module.getContext());
  for (auto op : rows) {
    rewriter.setInsertionPoint(op);
    if (failed(linalg::generalizeNamedOp(rewriter, op)))
      llvm::report_fatal_error("Cannot generalize a Poly row permutation");
  }
}
