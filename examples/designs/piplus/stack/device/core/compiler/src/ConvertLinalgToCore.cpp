//===- ConvertLinalgToCore.cpp - A graph on the piplus thread -------------===//
//
// convert-linalg-to-core places the work of a graph that the hardware thread of
// its core can do on that thread; the rest stays on the core CPU. The thread
// is the first of the core class (`core`, design.pb) with the Balls
// mxint8quant and mxint8mm, and takes:
//
//   - every fp32_to_mxint8 (mxint8quant);
//   - every linalg.matmul and linalg.batch_matmul whose two operands are
//     MXINT8 values, each decoded by a mxint8_to_fp32 and reaching the
//     product through transposes and reshapes only (mxint8mm). The decoding
//     and the transposes and reshapes go with the product: the Ball reads
//     the packed operands. A weight stored in the Ball's layout is decoded
//     from a slice of it, which goes with the product too.
//
// The placed operations that read each other's values form one subgraph, a
// function <function>_ht<n> of tensors: the unit the thread runs. Constants,
// empty tensors and fills of them are copied into the subgraph; the other
// values it reads are its inputs and those of its values the CPU reads its
// results. The schedule: a subgraph is launched as early as its inputs
// allow, right after the last value of the CPU it reads (or at the start of
// the graph), every input transferred to the thread before the first launch
// that reads it; every result is transferred back right before the first
// operation of the CPU that reads it, so the CPU goes on while the thread
// runs (the q, k and v projections of a layer start together). The core has
// one hardware thread, so no value passes between two threads.
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
      if (auto slice =
              op->getOperand(0).getDefiningOp<tensor::ExtractSliceOp>())
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

// Constants, empty tensors and fills of them: a subgraph makes its own.
bool isLocal(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (isa<arith::ConstantOp, tensor::EmptyOp>(op))
    return true;
  if (isa<linalg::FillOp>(op))
    return llvm::all_of(op->getOperands(), isLocal);
  return false;
}

// The values an operation reads: its operands and the values its regions
// read from outside.
SetVector<Value> readsOf(Operation *op) {
  SetVector<Value> values(op->operand_begin(), op->operand_end());
  getUsedValuesDefinedAbove(op->getRegions(), values);
  return values;
}

class ConvertLinalgToCorePass
    : public PassWrapper<ConvertLinalgToCorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToCorePass)

  ConvertLinalgToCorePass() = default;
  ConvertLinalgToCorePass(const ConvertLinalgToCorePass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "convert-linalg-to-core"; }
  StringRef getDescription() const final {
    return "Place the MXINT8 quantization and products of a graph on the "
           "hardware thread of its piplus core";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, CoreDialect,
                    func::FuncDialect, tensor::TensorDialect>();
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
    int64_t thread = -1;
    for (auto [local, index] : llvm::enumerate(placement->thread_indices())) {
      llvm::StringSet<> balls;
      for (const auto &ball : design.threads(index).balldomain().balls())
        balls.insert(ball.component());
      if (balls.contains("mxint8quant") && balls.contains("mxint8mm")) {
        thread = local;
        break;
      }
    }
    if (thread < 0)
      llvm::report_fatal_error(llvm::Twine("no thread of core class ") +
                               coreClass + " has mxint8quant and mxint8mm");

    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    IntegerAttr threadAttr =
        IntegerAttr::get(IntegerType::get(context, 64), thread);
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal())
        continue;
      Block &body = function.getBody().front();
      DenseSet<Operation *> placed;
      for (Operation &op : body) {
        if (isQuantize(&op))
          placed.insert(&op);
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        SmallVector<Operation *> lhs = decoding(op.getOperand(0));
        SmallVector<Operation *> rhs = decoding(op.getOperand(1));
        if (lhs.empty() || rhs.empty())
          continue;
        placed.insert(&op);
        placed.insert(lhs.begin(), lhs.end());
        placed.insert(rhs.begin(), rhs.end());
      }

      // The subgraphs: placed operations joined by their values, each in
      // the order of the block, with the last value of the CPU each reads.
      // A subgraph is launched as early as it can be, right after that
      // value (or at the start of the block); the subgraphs in the order of
      // their launches, then of their first operations.
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
      };
      SmallVector<Subgraph> subgraphs;
      for (auto &[leader, ops] : members) {
        DenseSet<Operation *> inside(ops.begin(), ops.end());
        Operation *latest = nullptr;
        for (Operation *op : ops)
          for (Value value : readsOf(op)) {
            Operation *def = value.getDefiningOp();
            if (!def || inside.contains(def) || isLocal(value))
              continue;
            def = body.findAncestorOpInBlock(*def);
            if (!latest || latest->isBeforeInBlock(def))
              latest = def;
          }
        subgraphs.push_back({latest, ops});
      }
      llvm::stable_sort(subgraphs, [](const Subgraph &a, const Subgraph &b) {
        if (a.latest == b.latest)
          return a.ops.front()->isBeforeInBlock(b.ops.front());
        return !a.latest || (b.latest && a.latest->isBeforeInBlock(b.latest));
      });

      // The transfer of each value of the CPU to the thread.
      DenseMap<Value, Value> sent;
      // The constants, empty tensors and fills subgraphs read.
      SetVector<Operation *> locals;
      std::function<void(Value)> collect = [&](Value value) {
        locals.insert(value.getDefiningOp());
        for (Value operand : value.getDefiningOp()->getOperands())
          collect(operand);
      };
      // The last launch after each last value of the CPU: the launches
      // after the same value follow each other.
      DenseMap<Operation *, Operation *> after;
      // The subgraphs follow their function, in order.
      Operation *previous = function;
      for (auto [number, subgraph] : llvm::enumerate(subgraphs)) {
        ArrayRef<Operation *> ops = subgraph.ops;
        DenseSet<Operation *> inside(ops.begin(), ops.end());
        auto isInside = [&](Operation *user) {
          return inside.contains(body.findAncestorOpInBlock(*user));
        };
        Operation *&cursor = after[subgraph.latest];
        if (!cursor)
          cursor = subgraph.latest;
        // The launch goes before `point`.
        Operation *point = cursor ? cursor->getNextNode() : &body.front();
        SetVector<Value> inputs, results;
        for (Operation *op : ops) {
          for (Value value : readsOf(op)) {
            if (inside.contains(value.getDefiningOp()))
              continue;
            if (isLocal(value))
              collect(value);
            else
              inputs.insert(value);
          }
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
                    ValueRange(inputs.getArrayRef()).getTypes(),
                    ValueRange(results.getArrayRef()).getTypes())),
                ValueRange(results.getArrayRef()).getTypes()));
        outlined.setPrivate();
        previous = outlined;
        builder.setInsertionPointToStart(outlined.addEntryBlock());
        IRMapping mapping;
        mapping.map(inputs.getArrayRef(),
                    outlined.getArguments().take_front(inputs.size()));
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
          returned.push_back(bufferization::MaterializeInDestinationOp::create(
                                 builder, loc, mapping.lookup(value),
                                 outlined.getArgument(inputs.size() + index))
                                 .getResult());
        func::ReturnOp::create(builder, loc, returned);

        // On the CPU: the transfers and the launch before `point`; each
        // result transferred back right before the first operation of the
        // CPU that reads it, so the CPU goes on while the thread runs.
        builder.setInsertionPoint(point);
        auto empty = [&](Value value) {
          auto type = cast<RankedTensorType>(value.getType());
          return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                         type.getElementType())
              .getResult();
        };
        SmallVector<Value> operands;
        for (Value value : inputs) {
          Value &copy = sent[value];
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
