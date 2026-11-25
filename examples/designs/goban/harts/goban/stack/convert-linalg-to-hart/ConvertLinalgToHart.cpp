//===- ConvertLinalgToHart.cpp - A graph on the matmul threads ------------===//
//
// convert-linalg-to-hart first compiles the FP32 row regions into operators
// of the GP domain (OutlineOperators.cpp), then places the work of a graph that
// the matmul threads of the goban hart can do on those threads; the rest stays
// on the hart CPU. The threads are those of the hart (design.pb, the device has
// one hart) with the Balls smatmul, toint8, int2fp and transpose, and take:
//
//   - every fp32_to_int8 (toint8);
//   - every int32_to_fp32 (int2fp);
//   - every bbacc.custom "row_operator", split by rows over
//     the threads, each part its own task;
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
// two threads for those INT8 subgraphs. Row operator tasks can consume another
// task's result; their logical dependencies are lowered to kernel transfers.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Hart/HartOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::hart;

LogicalResult outlineOperators(ModuleOp module);

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

class ConvertLinalgToHartPass
    : public PassWrapper<ConvertLinalgToHartPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToHartPass)

  ConvertLinalgToHartPass() = default;
  ConvertLinalgToHartPass(const ConvertLinalgToHartPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "convert-linalg-to-hart"; }
  StringRef getDescription() const final {
    return "Place INT8 computations and row operators on the hart's "
           "matmul threads";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::acc::AccDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect, HartDialect,
                    func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  Option<std::string> designPb{*this, "design",
                               llvm::cl::desc("design.pb of the device")};

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
    if (design.device().harts_size() != 1)
      llvm::report_fatal_error("goban has one hart");
    SmallVector<int64_t> threads;
    for (auto [local, index] :
         llvm::enumerate(design.device().harts(0).thread_indices())) {
      llvm::StringSet<> balls;
      for (const auto &ball : design.threads(index).balldomain().balls())
        balls.insert(ball.component());
      if (balls.contains("smatmul") && balls.contains("toint8") &&
          balls.contains("int2fp") && balls.contains("transpose"))
        threads.push_back(local);
    }
    if (threads.empty())
      llvm::report_fatal_error("no thread of the goban hart has smatmul, "
                               "toint8, int2fp and transpose");

    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    // In round-to-nearest FP32, exp(x) is zero below -104. Passing -inf
    // avoids newlib's underflow/errno path for masked attention scores.
    module.walk([&](math::ExpOp op) {
      if (!op.getType().isF32())
        return;
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value limit =
          arith::ConstantOp::create(b, loc, b.getF32FloatAttr(-104.0));
      Value infinity = arith::ConstantOp::create(
          b, loc,
          b.getFloatAttr(b.getF32Type(),
                         APFloat::getInf(APFloat::IEEEsingle(), true)));
      Value underflow = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                              op.getOperand(), limit);
      Value input =
          arith::SelectOp::create(b, loc, underflow, infinity, op.getOperand());
      op->setOperand(0, input);
    });
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal())
        continue;
      Block &body = function.getBody().front();

      // A row operator is independent of the other rows. Submit its partitions
      // before assembling the result so the kernel can run all local threads.
      for (auto op : llvm::to_vector(body.getOps<::buddy::acc::CustomOp>())) {
        if (op.getKind() != "row_operator")
          continue;
        auto type = cast<RankedTensorType>(op.getResult(0).getType());
        if (type.getRank() != 2 || type.getDimSize(0) < int64_t(threads.size()))
          continue;
        OpBuilder b(op);
        Location loc = op.getLoc();
        SmallVector<Value> parts;
        SmallVector<int64_t> starts, lengths;
        auto attr = [&](int64_t value) -> OpFoldResult {
          return b.getIndexAttr(value);
        };
        for (int64_t part = 0; part < int64_t(threads.size()); ++part) {
          int64_t start = type.getDimSize(0) * part / threads.size();
          int64_t length =
              type.getDimSize(0) * (part + 1) / threads.size() - start;
          SmallVector<Value> inputs;
          for (auto [input, mapAttr] : llvm::zip_equal(
                   op.getInputs(), op->getAttrOfType<ArrayAttr>("maps")
                                       .getValue()
                                       .drop_back())) {
            auto inputType = cast<RankedTensorType>(input.getType());
            auto map = cast<AffineMapAttr>(mapAttr).getValue();
            SmallVector<OpFoldResult> offsets(inputType.getRank(), attr(0)),
                strides(inputType.getRank(), attr(1)), sizes;
            bool sliced = false;
            for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
              bool row = isa<AffineDimExpr>(expression) &&
                         cast<AffineDimExpr>(expression).getPosition() == 0;
              offsets[axis] = attr(row ? start : 0);
              sizes.push_back(attr(row ? length : inputType.getDimSize(axis)));
              sliced |= row;
            }
            inputs.push_back(sliced
                                 ? tensor::ExtractSliceOp::create(
                                       b, loc, input, offsets, sizes, strides)
                                       .getResult()
                                 : input);
          }
          auto rows = ::buddy::acc::CustomOp::create(
              b, loc,
              TypeRange{RankedTensorType::get({length, type.getDimSize(1)},
                                              type.getElementType())},
              op.getKind(), inputs);
          rows->setDiscardableAttrs(op->getDiscardableAttrDictionary());
          parts.push_back(rows.getResult(0));
          starts.push_back(start);
          lengths.push_back(length);
        }
        Value result = tensor::EmptyOp::create(b, loc, type.getShape(),
                                               type.getElementType());
        for (auto [part, start, length] :
             llvm::zip_equal(parts, starts, lengths))
          result = tensor::InsertSliceOp::create(
              b, loc, part, result,
              ArrayRef<OpFoldResult>{attr(start), attr(0)},
              ArrayRef<OpFoldResult>{attr(length), attr(type.getDimSize(1))},
              ArrayRef<OpFoldResult>{attr(1), attr(1)});
        op.getResult(0).replaceAllUsesWith(result);
        op.erase();
      }

      // Each row partition owns its quantization and product. Clone the
      // offline weight views so sharing a parameter does not join the jobs.
      for (auto product : llvm::to_vector(body.getOps<linalg::MatmulOp>())) {
        auto type = cast<RankedTensorType>(product->getResult(0).getType());
        auto activation = product->getOperand(0)
                              .getDefiningOp<::buddy::frontend::Int8ToFP32Op>();
        if (!activation || !activation->hasOneUse() ||
            type.getDimSize(0) < int64_t(threads.size()) * 16)
          continue;
        auto quant = activation.getInput()
                         .getDefiningOp<::buddy::frontend::FP32ToInt8Op>();
        auto weight = decodeChain(product->getOperand(1));
        if (!quant || !quant->hasOneUse() || weight.empty())
          continue;
        auto multiplierType =
            cast<RankedTensorType>(quant.getMultiplier().getType());
        auto stepType = cast<RankedTensorType>(activation.getStep().getType());
        auto fill = product->getOperand(2).getDefiningOp<linalg::FillOp>();
        if (multiplierType.getRank() != 2 || stepType.getRank() != 2 || !fill)
          continue;
        auto decode = dyn_cast<::buddy::frontend::Int8ToFP32Op>(weight[0]);
        // The matmul may read a transpose of the decoded weight.
        for (Operation *operation : weight)
          if (auto candidate =
                  dyn_cast<::buddy::frontend::Int8ToFP32Op>(operation))
            decode = candidate;
        if (decode.getInput().getDefiningOp<::buddy::frontend::FP32ToInt8Op>())
          continue;
        OpBuilder b(product);
        Location loc = product.getLoc();
        auto attr = [&](int64_t value) -> OpFoldResult {
          return b.getIndexAttr(value);
        };
        auto slice = [&](Value value, int64_t first, int64_t length) -> Value {
          auto shape = cast<RankedTensorType>(value.getType());
          if (shape.getDimSize(0) == 1)
            return value;
          return tensor::ExtractSliceOp::create(
              b, loc, value, ArrayRef<OpFoldResult>{attr(first), attr(0)},
              ArrayRef<OpFoldResult>{attr(length), attr(shape.getDimSize(1))},
              ArrayRef<OpFoldResult>{attr(1), attr(1)});
        };
        int64_t rows = type.getDimSize(0), units = (rows + 15) / 16;
        SmallVector<Value> parts;
        SmallVector<int64_t> starts, lengths;
        for (int64_t part = 0; part < int64_t(threads.size()); ++part) {
          int64_t first = units * part / threads.size() * 16;
          int64_t end =
              std::min(rows, int64_t(units * (part + 1) / threads.size() * 16));
          int64_t length = end - first;
          IRMapping mapping;
          mapping.map(quant.getInput(), slice(quant.getInput(), first, length));
          mapping.map(quant.getMultiplier(),
                      slice(quant.getMultiplier(), first, length));
          Operation *codes = b.clone(*quant, mapping);
          auto inputType = cast<RankedTensorType>(quant.getInput().getType());
          codes->getResult(0).setType(RankedTensorType::get(
              {length, inputType.getDimSize(1)}, b.getI8Type()));
          mapping.map(activation.getStep(),
                      slice(activation.getStep(), first, length));
          Operation *values = b.clone(*activation, mapping);
          values->getResult(0).setType(RankedTensorType::get(
              {length, inputType.getDimSize(1)}, b.getF32Type()));
          for (Operation *operation : llvm::reverse(weight))
            b.clone(*operation, mapping);
          auto partType = RankedTensorType::get({length, type.getDimSize(1)},
                                                b.getF32Type());
          Value empty = tensor::EmptyOp::create(b, loc, partType.getShape(),
                                                b.getF32Type());
          Value initialized =
              linalg::FillOp::create(b, loc, fill.getDpsInputs(),
                                     ValueRange{empty})
                  .getResult(0);
          mapping.map(product->getOperand(2), initialized);
          Operation *task = b.clone(*product, mapping);
          task->getResult(0).setType(partType);
          parts.push_back(task->getResult(0));
          starts.push_back(first);
          lengths.push_back(length);
        }
        Value result = tensor::EmptyOp::create(b, loc, type.getShape(),
                                               type.getElementType());
        for (auto [part, first, length] :
             llvm::zip_equal(parts, starts, lengths))
          result = tensor::InsertSliceOp::create(
              b, loc, part, result,
              ArrayRef<OpFoldResult>{attr(first), attr(0)},
              ArrayRef<OpFoldResult>{attr(length), attr(type.getDimSize(1))},
              ArrayRef<OpFoldResult>{attr(1), attr(1)});
        product->getResult(0).replaceAllUsesWith(result);
        product.erase();
      }
      RewritePatternSet constantSlices(context);
      tensor::populateFoldConstantExtractSlicePatterns(
          constantSlices, [](tensor::ExtractSliceOp) { return true; });
      if (failed(applyPatternsGreedily(function, std::move(constantSlices))))
        return signalPassFailure();

      DenseSet<Operation *> placed, separate;
      for (Operation &op : body) {
        StringRef name = op.getName().getStringRef();
        if (auto custom = dyn_cast<::buddy::acc::CustomOp>(op);
            custom && custom.getKind() == "row_operator") {
          placed.insert(&op);
          separate.insert(&op);
        }
        if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
          auto type = cast<RankedTensorType>(transpose->getResult(0).getType());
          bool matrix = type.getRank() == 2 && type.getDimSize(0) % 4 == 0 &&
                        type.getDimSize(1) % 4 == 0;
          bool blocks =
              type.getRank() > 2 &&
              transpose.getPermutation().back() == type.getRank() - 1 &&
              type.getShape().back() % 4 == 0;
          if (type.hasStaticShape() && type.getElementType().isF32() &&
              (matrix || blocks) &&
              (matrix ? type.getDimSize(1) : type.getShape().back() / 4) <=
                  design.threads(design.device().harts(0).thread_indices(0))
                      .mem()
                      .bank()
                      .entries() &&
              decodeChain(transpose.getInput()).empty()) {
            placed.insert(&op);
            separate.insert(&op);
          }
        }
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
          if (placed.contains(value.getDefiningOp()) &&
              !separate.contains(&op) &&
              !separate.contains(value.getDefiningOp()))
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
        DenseSet<Value> visited;
        Operation *latest = nullptr;
        std::function<void(Value)> findHostInput = [&](Value value) {
          if (!visited.insert(value).second)
            return;
          Operation *def = value.getDefiningOp();
          if (!def || inside.contains(def) || isLocal(value))
            return;
          if (placed.contains(def)) {
            for (Value input : def->getOperands())
              findHostInput(input);
          } else if (!latest || latest->isBeforeInBlock(def)) {
            latest = def;
          }
        };
        for (Operation *op : ops)
          for (Value value : op->getOperands())
            findHostInput(value);
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
        for (Value value : inputs)
          if (Operation *def = value.getDefiningOp();
              def && def->getBlock() == &body &&
              (point == def || point->isBeforeInBlock(def)))
            point = def->getNextNode();
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
        outlined->setAttr("hart.program_bank",
                          builder.getI64IntegerAttr(
                              design.device().harts(0).virtual_bank_count()));
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
        // The launch computes its results here: their locations.
        SmallVector<Location> computed;
        for (Value value : results)
          computed.push_back(value.getDefiningOp()->getLoc());
        auto launch =
            LaunchOp::create(builder, FusedLoc::get(context, computed),
                             ValueRange(results.getArrayRef()).getTypes(),
                             FlatSymbolRefAttr::get(outlined.getNameAttr()),
                             sent, outs, threadAttr);
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

PassRegistration<ConvertLinalgToHartPass> registration;
} // namespace
