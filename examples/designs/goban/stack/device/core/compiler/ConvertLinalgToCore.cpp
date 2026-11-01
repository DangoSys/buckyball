//===- ConvertLinalgToCore.cpp - A graph on the matmul threads ------------===//
//
// convert-linalg-to-core places the work of a graph that the matmul threads of
// the goban core can do on those threads; the rest stays on the core CPU. The
// threads are those of the core (design.pb, the device has one core) with
// the Balls smatmul, toint8, int2fp and transpose, and take:
//
//   - every fp32_to_int8 (toint8);
//   - every int32_to_fp32 (int2fp);
//   - every linalg.matmul and linalg.batch_matmul whose two operands are INT8
//     values, each decoded by an int8_to_fp32 and reaching the product
//     through transposes and reshapes only (smatmul, transpose). The
//     decoding and the transposes and reshapes go with the product: the
//     Balls read the INT8 operands.
//
// Placed operations that read each other's values form one subgraph,
// <function>_ht<n>, a private function of tensors: the unit a thread runs.
// Constants, empty tensors and fills of them are cloned into it; the other
// values it reads are its inputs, its values the CPU reads are its results.
//
// The schedule: a subgraph is launched as early as its inputs allow, right
// after the last CPU value it reads (or at the start of the graph), each
// input transferred to its thread right before the launch; each result is
// transferred back right before the first CPU operation that reads it. The
// subgraphs take the matmul threads in turn, in the order of their launches,
// so subgraphs that read no value of each other (the quantized query and key
// products of an attention, the bias of a layer and its product) are on
// different threads. No value passes between two subgraphs, hence between
// two threads: the CPU reads each result.
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
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::core;

namespace {
// The chain from an operand of a product back to the int8_to_fp32 that
// decodes it, through transposes and reshapes; empty if there is none. A
// weight stored in the smatmul layout reaches the decoding through slices,
// reshapes and a transpose: they go with it, smatmul reads the parameter.
SmallVector<Operation *> decodeChain(Value value) {
  SmallVector<Operation *> chain;
  while (Operation *op = value.getDefiningOp()) {
    chain.push_back(op);
    if (op->getName().getStringRef() == "frontend.int8_to_fp32") {
      Value input = op->getOperand(0);
      while (Operation *view = input.getDefiningOp()) {
        if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
                 tensor::ExpandShapeOp, tensor::ExtractSliceOp>(view))
          break;
        chain.push_back(view);
        input = view->getOperand(0);
      }
      return chain;
    }
    if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp>(op))
      return {};
    value = op->getOperand(0);
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
  return isa<linalg::FillOp>(op) && llvm::all_of(op->getOperands(), isLocal);
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
    return "Place the INT8 quantization, decoding and products of a graph on "
           "a matmul thread of the goban core";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, CoreDialect,
                    func::FuncDialect, tensor::TensorDialect>();
  }

  Option<std::string> designPb{*this, "design",
                               llvm::cl::desc("design.pb of the device")};

  void runOnOperation() override {
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    if (!buffer)
      llvm::report_fatal_error(llvm::Twine("cannot read ") + designPb);
    buckyball::config::Design design;
    if (!design.ParseFromArray((*buffer)->getBufferStart(),
                               (*buffer)->getBufferSize()))
      llvm::report_fatal_error(llvm::Twine("cannot parse ") + designPb);
    if (design.device().cores_size() != 1)
      llvm::report_fatal_error("goban has one core");
    SmallVector<int64_t> threads;
    for (auto [local, index] :
         llvm::enumerate(design.device().cores(0).thread_indices())) {
      llvm::StringSet<> balls;
      for (const auto &ball : design.threads(index).balldomain().balls())
        balls.insert(ball.component());
      if (balls.contains("smatmul") && balls.contains("toint8") &&
          balls.contains("int2fp") && balls.contains("transpose"))
        threads.push_back(local);
    }
    if (threads.empty())
      llvm::report_fatal_error("no thread of the goban core has smatmul, "
                               "toint8, int2fp and transpose");

    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal())
        continue;
      Block &body = function.getBody().front();

      DenseSet<Operation *> placed;
      for (Operation &op : body) {
        StringRef name = op.getName().getStringRef();
        if (name == "frontend.fp32_to_int8" || name == "frontend.int32_to_fp32")
          placed.insert(&op);
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        SmallVector<Operation *> lhs = decodeChain(op.getOperand(0));
        SmallVector<Operation *> rhs = decodeChain(op.getOperand(1));
        if (lhs.empty() || rhs.empty())
          continue;
        placed.insert(&op);
        placed.insert(lhs.begin(), lhs.end());
        placed.insert(rhs.begin(), rhs.end());
      }

      // Subgraphs: placed operations joined by the values they share, each
      // in block order, with the last CPU value each reads; the subgraphs in
      // the order of their launches, then of their first operations.
      llvm::EquivalenceClasses<Operation *> joined;
      for (Operation &op : body) {
        if (!placed.contains(&op))
          continue;
        joined.insert(&op);
        for (Value value : op.getOperands())
          if (placed.contains(value.getDefiningOp()))
            joined.unionSets(&op, value.getDefiningOp());
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
          for (Value value : op->getOperands()) {
            Operation *def = value.getDefiningOp();
            if (!def || inside.contains(def) || isLocal(value))
              continue;
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

      // The constants, empty tensors and fills subgraphs read, with what
      // they read.
      SetVector<Operation *> locals;
      std::function<void(Value)> collect = [&](Value value) {
        locals.insert(value.getDefiningOp());
        for (Value operand : value.getDefiningOp()->getOperands())
          collect(operand);
      };
      // The last launch after each last CPU value: the launches after the
      // same value follow each other.
      DenseMap<Operation *, Operation *> after;
      Operation *previous = function;
      for (auto [number, subgraph] : llvm::enumerate(subgraphs)) {
        ArrayRef<Operation *> ops = subgraph.ops;
        DenseSet<Operation *> inside(ops.begin(), ops.end());
        IntegerAttr threadAttr = IntegerAttr::get(
            IntegerType::get(context, 64), threads[number % threads.size()]);
        Operation *&cursor = after[subgraph.latest];
        if (!cursor)
          cursor = subgraph.latest;
        // The launch goes before `point`.
        Operation *point = cursor ? cursor->getNextNode() : &body.front();
        SetVector<Value> inputs, results;
        for (Operation *op : ops) {
          for (Value value : op->getOperands()) {
            if (inside.contains(value.getDefiningOp()))
              continue;
            if (isLocal(value)) {
              collect(value);
              continue;
            }
            inputs.insert(value);
          }
          for (Value value : op->getResults())
            for (Operation *user : value.getUsers())
              if (!inside.contains(body.findAncestorOpInBlock(*user)))
                results.insert(value);
        }
        for (Value value : results)
          for (Operation *user : value.getUsers()) {
            Operation *top = body.findAncestorOpInBlock(*user);
            if (!inside.contains(top) && top->isBeforeInBlock(point)) {
              user->emitError("reads a value of a thread subgraph before its "
                              "last input is computed on the CPU");
              return signalPassFailure();
            }
          }

        // The subgraph, after the function or the subgraph before it.
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
          for (Value operand : value.getDefiningOp()->getOperands())
            materialize(operand);
          builder.clone(*value.getDefiningOp(), mapping);
        };
        for (Operation *op : ops) {
          for (Value value : op->getOperands())
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

        // On the CPU: transfers to the thread and the launch before `point`,
        // each result transferred back before the first CPU operation that
        // reads it.
        builder.setInsertionPoint(point);
        auto empty = [&](Value value) -> Value {
          auto type = cast<RankedTensorType>(value.getType());
          return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                         type.getElementType());
        };
        SmallVector<Value> sent, outs, backs;
        for (Value value : inputs)
          sent.push_back(
              TransferOp::create(builder, loc, TypeRange{value.getType()},
                                 value, empty(value), IntegerAttr(), threadAttr)
                  .getResult()
                  .front());
        for (Value value : results) {
          outs.push_back(empty(value));
          backs.push_back(empty(value));
        }
        auto launch = LaunchOp::create(
            builder, loc, ValueRange(results.getArrayRef()).getTypes(),
            FlatSymbolRefAttr::get(outlined.getNameAttr()), sent, outs,
            threadAttr);
        cursor = launch;
        for (auto [index, result] : llvm::enumerate(launch.getResults())) {
          Value value = results[index];
          Operation *first = nullptr;
          for (Operation *user : value.getUsers()) {
            Operation *top = body.findAncestorOpInBlock(*user);
            if (inside.contains(top))
              continue;
            if (!first || top->isBeforeInBlock(first))
              first = top;
          }
          builder.setInsertionPoint(first);
          Value back = TransferOp::create(
                           builder, loc, TypeRange{value.getType()}, result,
                           backs[index], threadAttr, IntegerAttr())
                           .getResult()
                           .front();
          value.replaceUsesWithIf(back, [&](OpOperand &use) {
            return !inside.contains(
                body.findAncestorOpInBlock(*use.getOwner()));
          });
        }
        for (Operation *op : llvm::reverse(ops))
          op->erase();
      }

      // The constants, empty tensors and fills only subgraphs read: users
      // first.
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
