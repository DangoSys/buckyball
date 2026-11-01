//===- ConvertLinalgToCore.cpp - A graph on the poly threads --------------===//
//
// convert-linalg-to-core places the work of a segment graph that the hardware
// threads of a poly core can do on those threads; the rest stays on the core
// CPU. A poly core has attention and ffn threads (design.pb, by component),
// each with MXINT8 quantization/matrix Balls and mxmm. A thread takes:
//
//   - every fp32_to_mxint8 (mxint8quantize) but one only the operands of
//     FP32 products read;
//   - every linalg.matmul and linalg.batch_matmul (mxint8matmul when both
//     operands are MXINT8, else mxmm in FP32). An operand of an MXINT8
//     product that is an MXINT8 value decoded by a mxint8_to_fp32 and
//     reaching the product through transposes and reshapes only goes with
//     the product, that decoding, the slice it decodes (a weight padded to
//     the Ball layout) and those transposes and reshapes included: the Ball
//     reads the packed operand. The decoded operand of an FP32 product stays
//     on the CPU, which decodes it in software; the thread reads FP32.
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

#include "Core/CoreOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::core;

namespace {
bool isQuantize(Operation *op) {
  return op->getName().getStringRef() == "frontend.fp32_to_mxint8";
}

bool isDecode(Operation *op) {
  return op->getName().getStringRef() == "frontend.mxint8_to_fp32";
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
    else if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op))
      value = op->getOperand(0);
    else
      break;
  }
  return {};
}

// Constants, empty tensors, fills of them and sizes: a subgraph makes its
// own.
bool isLocal(Operation *op) {
  if (isa<arith::ConstantOp, tensor::EmptyOp, tensor::DimOp>(op))
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
    if (def->getDialect()->getNamespace() != "arith") {
      def->emitError("a size computed by an unknown operation");
      llvm::report_fatal_error(
          "convert-linalg-to-core: a size it cannot compute");
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
      for (int64_t source : collapse.getReassociationIndices()[dim])
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
        "convert-linalg-to-core: a size it cannot compute");
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

class ConvertLinalgToCorePass
    : public PassWrapper<ConvertLinalgToCorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToCorePass)

  ConvertLinalgToCorePass() = default;
  ConvertLinalgToCorePass(const ConvertLinalgToCorePass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "convert-linalg-to-core"; }
  StringRef getDescription() const final {
    return "Place the MXINT8 quantizations and matrix products of a graph "
           "on the attention or ffn threads of its poly core";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, CoreDialect,
                    arith::ArithDialect, func::FuncDialect,
                    tensor::TensorDialect>();
  }

  Option<std::string> designPb{*this, "design",
                               llvm::cl::desc("design.pb of the device")};
  Option<std::string> coreClass{
      *this, "core", llvm::cl::desc("class of the core the graph runs on")};

  void runOnOperation() override {
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    if (!buffer)
      llvm::report_fatal_error(llvm::Twine("cannot read ") + designPb);
    buckyball::config::Design design;
    if (!design.ParseFromArray((*buffer)->getBufferStart(),
                               (*buffer)->getBufferSize()))
      llvm::report_fatal_error(llvm::Twine("cannot parse ") + designPb);
    const buckyball::config::CorePlacement *placement = nullptr;
    for (const auto &core : design.device().cores())
      if (core.component() == coreClass) {
        placement = &core;
        break;
      }
    if (!placement)
      llvm::report_fatal_error(llvm::Twine("no core of class ") + coreClass);
    // The threads of each kind, by their index in the core.
    llvm::StringMap<SmallVector<int64_t>> kinds;
    for (auto [local, index] : llvm::enumerate(placement->thread_indices())) {
      const auto &thread = design.threads(index);
      llvm::StringSet<> balls;
      for (const auto &ball : thread.balldomain().balls())
        balls.insert(ball.component());
      if (!balls.contains("mxint8quantize") ||
          !balls.contains("mxint8matmul") || !balls.contains("mxmm"))
        llvm::report_fatal_error(llvm::Twine("thread ") + thread.role() +
                                 " of core class " + coreClass +
                                 " lacks mxint8quantize, mxint8matmul or mxmm");
      kinds[thread.pkg()].push_back(local);
    }
    if (!kinds.contains("attention") || !kinds.contains("ffn"))
      llvm::report_fatal_error(llvm::Twine("core class ") + coreClass +
                               " has no attention or no ffn thread");

    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal())
        continue;
      Block &body = function.getBody().front();
      DenseSet<Operation *> placed;
      // The decodings of the operands of FP32 products (one operand not
      // MXINT8): the CPU decodes them, and quantizes what only they read.
      DenseSet<Operation *> software;
      bool attention = false;
      for (Operation &op : body) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        attention |= isa<linalg::BatchMatmulOp>(op);
        placed.insert(&op);
        SmallVector<Operation *> lhs = decoding(op.getOperand(0)),
                                 rhs = decoding(op.getOperand(1));
        if (lhs.empty() || rhs.empty()) {
          for (auto *chain : {&lhs, &rhs})
            software.insert(chain->begin(), chain->end());
          continue;
        }
        placed.insert(lhs.begin(), lhs.end());
        placed.insert(rhs.begin(), rhs.end());
      }
      for (Operation &op : body)
        if (isQuantize(&op) &&
            !llvm::all_of(op.getUsers(), [&](Operation *user) {
              return software.contains(user);
            }))
          placed.insert(&op);
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
          if (Operation *def = value.getDefiningOp(); placed.contains(def))
            joined.unionSets(&op, def);
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
        auto launch = LaunchOp::create(
            builder, loc, ValueRange(results.getArrayRef()).getTypes(),
            FlatSymbolRefAttr::get(outlined.getNameAttr()), operands, outs,
            threadAttr);
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

PassRegistration<ConvertLinalgToCorePass> registration;
} // namespace
