//===- ConvertLinalgToHart.cpp - A graph on the piplus thread -------------===//
//
// convert-linalg-to-hart first gives each FP32 row softmax to SoftmaxBall and
// compiles the other FP32 row regions into operators of the GP domain
// (OutlineSoftmax.cpp, OutlineOperators.cpp), then places the work of a graph
// that the hardware thread of its hart can do on that thread; the rest stays
// on the hart CPU. The thread
// is the first of the hart class (`hart`, design.pb) with the Balls
// mxint8quant and mxint8mm, and takes:
//
//   - every fp32_to_mxint8 (mxint8quant);
//   - every linalg.matmul and linalg.batch_matmul whose two operands are
//     MXINT8 values, each decoded by a mxint8_to_fp32 and reaching the
//     product through transposes and reshapes only (mxint8mm). The decoding
//     and the transposes and reshapes go with the product: the Ball reads
//     the packed operands. A weight stored in the Ball's layout is decoded
//     from a slice of it, which goes with the product too;
//   - every bbacc.custom "row_operator", "softmax" and "layernorm", its own
//     subgraph:
//     the GP domain of the thread computes it, its operators in the program
//     bank after the virtual banks;
//   - every bbacc.custom "attention" (OutlineAttention.cpp), with the
//     decodings of its operands as a product and the views their
//     quantizations read through.
//
// The placed operations that read each other's values form one subgraph, a
// function <function>_ht<n> of tensors: the unit the thread runs. Constants,
// empty tensors and fills of them are copied into the subgraph; the other
// values it reads are its inputs and those of its values the CPU reads its
// results. The schedule: a subgraph is launched as early as its inputs
// allow, right after the last value of the CPU it reads (or at the start of
// the graph), every input transferred to the thread right before each
// launch that reads it; every result is transferred back right before the first
// operation of the CPU that reads it, so the CPU goes on while the thread
// runs (the q, k and v projections of a layer start together). With several
// threads the large row work is split by rows over them first
// (SplitRows.cpp), and each subgraph takes the thread that starts it first
// (its inputs there, the thread free); no value passes between two
// subgraphs, hence between two threads. Subgraphs equal but for their
// locations (the layers) are one function, launched by each.
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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::hart;

void quantizeBeforeViews(ModuleOp module);
void fuseBias(ModuleOp module);
void splitRows(ModuleOp module, int64_t threads);
LogicalResult outlineOperators(ModuleOp module,
                               const buckyball::config::Design &design,
                               StringRef hartClass);

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

class ConvertLinalgToHartPass
    : public PassWrapper<ConvertLinalgToHartPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToHartPass)

  ConvertLinalgToHartPass() = default;
  ConvertLinalgToHartPass(const ConvertLinalgToHartPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "convert-linalg-to-hart"; }
  StringRef getDescription() const final {
    return "Place the MXINT8 quantization and products of a graph on the "
           "hardware thread of its piplus hart";
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
    quantizeBeforeViews(getOperation());
    fuseBias(getOperation());
    if (failed(outlineOperators(getOperation(), design, hartClass)))
      return signalPassFailure();
    const buckyball::config::HartPlacement *placement = nullptr;
    for (const auto &hart : design.device().harts())
      if (hart.component() == hartClass) {
        placement = &hart;
        break;
      }
    if (!placement)
      llvm::report_fatal_error(llvm::Twine("no hart of class ") + hartClass);
    SmallVector<int64_t> threads;
    for (auto [local, index] : llvm::enumerate(placement->thread_indices())) {
      llvm::StringSet<> balls;
      for (const auto &ball : design.threads(index).balldomain().balls())
        balls.insert(ball.component());
      if (balls.contains("mxint8quant") && balls.contains("mxint8mm"))
        threads.push_back(local);
    }
    if (threads.empty())
      llvm::report_fatal_error(llvm::Twine("no thread of hart class ") +
                               hartClass + " has mxint8quant and mxint8mm");

    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    splitRows(module, threads.size());
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
      DenseSet<Operation *> placed, separate;
      for (Operation &op : body) {
        if (isQuantize(&op))
          placed.insert(&op);
        if (auto custom = dyn_cast<::buddy::acc::CustomOp>(op);
            custom && (custom.getKind() == "row_operator" ||
                       custom.getKind() == "softmax" ||
                       custom.getKind() == "layernorm")) {
          placed.insert(&op);
          separate.insert(&op);
        }
        if (auto custom = dyn_cast<::buddy::acc::CustomOp>(op);
            custom && custom.getKind() == "attention") {
          placed.insert(&op);
          for (Value input :
               custom.getInputs().drop_back(custom->hasAttr("masked"))) {
            SmallVector<Operation *> chain = decoding(input);
            placed.insert(chain.begin(), chain.end());
            // The views its quantization reads through (v's heads, which
            // it quantizes along the keys): the thread's DMA reads them in
            // place.
            for (Operation *op : chain) {
              if (!isDecode(op))
                continue;
              Operation *view = op->getOperand(0).getDefiningOp();
              if (view && isQuantize(view))
                view = view->getOperand(0).getDefiningOp();
              while (
                  isa_and_nonnull<linalg::TransposeOp, tensor::CollapseShapeOp,
                                  tensor::ExpandShapeOp>(view)) {
                placed.insert(view);
                view = view->getOperand(0).getDefiningOp();
              }
            }
          }
        }
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          continue;
        SmallVector<Operation *> lhs = decoding(op.getOperand(0));
        SmallVector<Operation *> rhs = decoding(op.getOperand(1));
        if (lhs.empty() || rhs.empty())
          continue;
        placed.insert(&op);
        placed.insert(lhs.begin(), lhs.end());
        placed.insert(rhs.begin(), rhs.end());
        // The start of the sums (FuseBias.cpp): a bias broadcast, a
        // residual read through a reshape, or their sum.
        Value start = op.getOperand(2);
        if (auto sum = start.getDefiningOp<linalg::GenericOp>();
            sum && sum.getNumDpsInputs() == 2 &&
            sum.getDpsInputs()[1].getDefiningOp<linalg::BroadcastOp>()) {
          placed.insert(sum);
          start = sum.getDpsInputs()[0];
          placed.insert(sum.getDpsInputs()[1].getDefiningOp());
        }
        if (isa_and_nonnull<linalg::BroadcastOp, tensor::CollapseShapeOp>(
                start.getDefiningOp()))
          placed.insert(start.getDefiningOp());
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
        // A product read as the start of another product's sums (the
        // residual, FuseBias.cpp), not as an earlier term of the same A, is
        // an input of that product's subgraph: the CPU may read it between
        // the two (the layer normalization of the residual stream). So is a
        // product another subgraph quantizes (the q and k of the attention
        // scores): a subgraph is one recorded stream of the thread, so the
        // projections and the attention products go to their own.
        auto apart = [&](Operation *user, Value value) {
          auto product = value.getDefiningOp<linalg::MatmulOp>();
          if (!product)
            return false;
          if (isQuantize(user))
            return true;
          if (auto later = dyn_cast<linalg::MatmulOp>(user))
            return value == later.getOutputs()[0] &&
                   later.getInputs()[0] != product.getInputs()[0];
          return user->hasOneUse() &&
                 isa<linalg::MatmulOp>(*user->getUsers().begin()) &&
                 user->getUses().begin()->getOperandNumber() == 2;
        };
        for (Value value : op.getOperands())
          if (Operation *def = value.getDefiningOp();
              placed.contains(def) && !separate.contains(&op) &&
              !separate.contains(def) && !apart(&op, value))
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
        DenseSet<Value> visited;
        Operation *latest = nullptr;
        // Through the subgraphs it reads, to the values of the CPU.
        std::function<void(Value)> findHostInput = [&](Value value) {
          if (!visited.insert(value).second)
            return;
          Operation *def = value.getDefiningOp();
          if (!def || inside.contains(def) || isLocal(value))
            return;
          def = body.findAncestorOpInBlock(*def);
          if (placed.contains(def)) {
            for (Value input : readsOf(def))
              findHostInput(input);
          } else if (!latest || latest->isBeforeInBlock(def)) {
            latest = def;
          }
        };
        for (Operation *op : ops)
          for (Value value : readsOf(op))
            findHostInput(value);
        subgraphs.push_back({latest, ops});
      }
      llvm::stable_sort(subgraphs, [](const Subgraph &a, const Subgraph &b) {
        if (a.latest == b.latest)
          return a.ops.front()->isBeforeInBlock(b.ops.front());
        return !a.latest || (b.latest && a.latest->isBeforeInBlock(b.latest));
      });
      // A subgraph follows the subgraphs whose values it reads (the bias of
      // an attention, computed after the same value of the CPU).
      for (size_t i = 0; i < subgraphs.size(); ++i)
        for (size_t j = i + 1; j < subgraphs.size(); ++j)
          if (llvm::any_of(subgraphs[i].ops, [&](Operation *op) {
                return llvm::any_of(op->getOperands(), [&](Value value) {
                  return llvm::is_contained(subgraphs[j].ops,
                                            value.getDefiningOp());
                });
              })) {
            Subgraph producer = subgraphs[j];
            subgraphs.erase(subgraphs.begin() + j);
            subgraphs.insert(subgraphs.begin() + i, producer);
            --i;
            break;
          }

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
      // When each thread is free and when the values of the placed
      // operations are there, in cycles of the Balls and the GP domain: a
      // product's multiply-accumulates over the 256 of the array a cycle,
      // any other computation's elements four a cycle.
      SmallVector<int64_t> free(threads.size(), 0);
      int64_t ready = 0;
      DenseMap<Operation *, int64_t> ends;
      DenseMap<Operation *, int64_t> known;
      // When a value is there: the end of the launch that computes it, or
      // of the latest one the CPU operations before it read (the subgraphs
      // a subgraph reads were launched before it).
      std::function<int64_t(Value)> readyAt = [&](Value value) -> int64_t {
        Operation *def = value.getDefiningOp();
        if (!def || def->getBlock() != &body)
          return 0;
        if (auto end = ends.find(def); end != ends.end())
          return end->second;
        auto [found, inserted] = known.try_emplace(def, 0);
        if (!inserted)
          return found->second;
        int64_t ready = 0;
        for (Value operand : def->getOperands())
          ready = std::max(ready, readyAt(operand));
        return known[def] = ready;
      };
      for (auto [number, subgraph] : llvm::enumerate(subgraphs)) {
        // Each subgraph, in the order of the launches, takes the thread that
        // starts it first, once its inputs are there and the thread is free
        // (of those, the thread free first): the row parts of a product
        // (SplitRows.cpp) and subgraphs that read no value of each other
        // run together on different threads.
        int64_t cost = 0;
        for (Operation *op : subgraph.ops) {
          auto type =
              op->getNumResults()
                  ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                  : RankedTensorType();
          if (!type || !type.hasStaticShape())
            continue;
          if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
            cost += type.getNumElements() *
                    cast<RankedTensorType>(op->getOperand(0).getType())
                        .getShape()
                        .back() /
                    256;
          else if (auto custom = dyn_cast<::buddy::acc::CustomOp>(op);
                   custom && custom.getKind() == "attention")
            // Its two products over the T keys of its parts, as many
            // multiply-accumulates.
            for (int64_t part = 0;
                 part < custom->getAttrOfType<IntegerAttr>("segments").getInt();
                 ++part)
              cost += 2 * type.getNumElements() *
                      cast<RankedTensorType>(op->getOperand(1 + part).getType())
                          .getShape()
                          .back() /
                      256;
          else if (isa<::buddy::acc::CustomOp, linalg::GenericOp>(op) ||
                   isQuantize(op))
            cost += type.getNumElements() / 4;
        }
        // The CPU launches in order: it waited for the inputs of the
        // launches before this one.
        for (Operation *op : subgraph.ops)
          for (Value value : op->getOperands())
            if (!llvm::is_contained(subgraph.ops, value.getDefiningOp()))
              ready = std::max(ready, readyAt(value));
        int64_t least = 0;
        for (size_t thread = 1; thread < free.size(); ++thread)
          if (std::pair(std::max(free[thread], ready), free[thread]) <
              std::pair(std::max(free[least], ready), free[least]))
            least = thread;
        free[least] = std::max(free[least], ready) + cost;
        IntegerAttr threadAttr =
            IntegerAttr::get(IntegerType::get(context, 64), threads[least]);
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
            for (Operation *user : value.getUsers())
              if (!isInside(user))
                results.insert(value);
        }
        // After the results of the subgraphs it reads, transferred back.
        for (Value value : inputs)
          if (Operation *def = value.getDefiningOp();
              def && def->getBlock() == &body &&
              (point == def || point->isBeforeInBlock(def)))
            point = def->getNextNode();
        for (Value value : results)
          for (Operation *user : value.getUsers())
            if (!isInside(user) &&
                body.findAncestorOpInBlock(*user)->isBeforeInBlock(point)) {
              user->emitError("reads a value of a thread subgraph before "
                              "the subgraph is launched");
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
                    ValueRange(inputs.getArrayRef()).getTypes(),
                    ValueRange(results.getArrayRef()).getTypes())),
                ValueRange(results.getArrayRef()).getTypes()));
        outlined.setPrivate();
        outlined->setAttr(
            "hart.program_bank",
            builder.getI64IntegerAttr(placement->virtual_bank_count()));
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
        // An input that is a slice of a larger value of the CPU (some heads
        // of every token) and that only operators, softmax, layer
        // normalizations and online quantizations (through views or not)
        // read, row by row, is read
        // in place: its argument takes that value's strides, and lower-hart
        // passes the slice.
        for (auto [index, value] : llvm::enumerate(inputs)) {
          // A view on the way to a quantization (v's heads of an
          // attention) is read as the quantization reads.
          std::function<bool(Operation *)> inPlace = [&](Operation *user) {
            if (isa<::buddy::frontend::FP32ToMXINT8Op>(user))
              return true;
            if (isa<linalg::TransposeOp, tensor::CollapseShapeOp,
                    tensor::ExpandShapeOp>(user))
              return llvm::all_of(user->getUsers(), inPlace);
            auto custom = dyn_cast<::buddy::acc::CustomOp>(user);
            return custom && (custom.getKind() == "row_operator" ||
                              custom.getKind() == "softmax" ||
                              custom.getKind() == "layernorm");
          };
          bool readers =
              llvm::all_of(outlined.getArgument(index).getUsers(), inPlace);
          // An argument of the graph the device passes in place (a rank's
          // slice of a state) keeps its layout.
          if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto graph = cast<func::FuncOp>(argument.getOwner()->getParentOp());
            if (auto layout = graph.getArgAttrOfType<StridedLayoutAttr>(
                    argument.getArgNumber(), "bufferization.buffer_layout");
                layout && readers)
              outlined.setArgAttr(index, "bufferization.buffer_layout", layout);
            continue;
          }
          // The slice, or the slice whose axes that are not one the value
          // is (the rows of a mask).
          Value sliced = value;
          while (Operation *def = sliced.getDefiningOp()) {
            if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(def))
              break;
            sliced = def->getOperand(0);
          }
          auto slice = sliced.getDefiningOp<tensor::ExtractSliceOp>();
          if (!slice || !slice.hasUnitStride() ||
              !slice.getSourceType().hasStaticShape() ||
              slice.getSourceType().getRank() != slice.getType().getRank() ||
              !readers)
            continue;
          SmallVector<int64_t> kept;
          for (int64_t axis = 0; axis < slice.getType().getRank(); ++axis)
            if (slice.getType().getDimSize(axis) != 1)
              kept.push_back(axis);
          auto valueType = cast<RankedTensorType>(value.getType());
          bool same = int64_t(kept.size()) == valueType.getRank();
          for (int64_t i = 0; same && i < valueType.getRank(); ++i)
            same =
                slice.getType().getDimSize(kept[i]) == valueType.getDimSize(i);
          if (sliced != value && !same)
            continue;
          // The strides of the sliced value: row-major, or those of the
          // graph's argument it is a slice of.
          ArrayRef<int64_t> shape = slice.getSourceType().getShape();
          SmallVector<int64_t> strides(shape.size());
          int64_t stride = 1;
          for (int64_t axis = shape.size() - 1; axis >= 0; --axis) {
            strides[axis] = stride;
            stride *= shape[axis];
          }
          if (auto argument = dyn_cast<BlockArgument>(slice.getSource()))
            if (auto layout =
                    cast<func::FuncOp>(argument.getOwner()->getParentOp())
                        .getArgAttrOfType<StridedLayoutAttr>(
                            argument.getArgNumber(),
                            "bufferization.buffer_layout"))
              strides.assign(layout.getStrides().begin(),
                             layout.getStrides().end());
          if (sliced != value) {
            SmallVector<int64_t> all = strides;
            strides.clear();
            for (int64_t axis : kept)
              strides.push_back(all[axis]);
          }
          outlined.setArgAttr(
              index, "bufferization.buffer_layout",
              StridedLayoutAttr::get(context, ShapedType::kDynamic, strides));
        }

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
        // A transfer per launch: the thread reads the CPU's buffer in place
        // (lower-hart), which must hold the value until each launch.
        for (Value value : inputs)
          operands.push_back(
              TransferOp::create(builder, loc, TypeRange{value.getType()},
                                 value, empty(value), IntegerAttr(), threadAttr)
                  .getResult()
                  .front());
        SmallVector<Value> outs, backs;
        for (Value value : results) {
          outs.push_back(empty(value));
          backs.push_back(empty(value));
        }
        // The launch computes its results here: their locations, which a
        // subgraph shared with other layers (below) does not keep.
        SmallVector<Location> computed;
        for (Value value : results)
          computed.push_back(value.getDefiningOp()->getLoc());
        auto launch =
            LaunchOp::create(builder, FusedLoc::get(context, computed),
                             ValueRange(results.getArrayRef()).getTypes(),
                             FlatSymbolRefAttr::get(outlined.getNameAttr()),
                             operands, outs, threadAttr);
        cursor = launch;
        ends[launch] = free[least];
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
          // A result inserted through a reshape into a larger value (the
          // heads of a step's own keys after the prefix's): it comes back
          // into its place, the destination slice reshaped to the result.
          auto expand = dyn_cast<tensor::ExpandShapeOp>(first);
          auto through =
              expand && expand.getSrc() == value && expand->hasOneUse()
                  ? dyn_cast<tensor::InsertSliceOp>(*expand->getUsers().begin())
                  : tensor::InsertSliceOp();
          if (through && through.getSource() == expand.getResult() &&
              llvm::all_of(value.getUsers(), [&](Operation *user) {
                return isInside(user) || user == expand;
              })) {
            builder.setInsertionPoint(through);
            Value slice = tensor::ExtractSliceOp::create(
                builder, loc, expand.getResultType(), through.getDest(),
                through.getMixedOffsets(), through.getMixedSizes(),
                through.getMixedStrides());
            Value place = tensor::CollapseShapeOp::create(
                builder, loc, cast<RankedTensorType>(value.getType()), slice,
                expand.getReassociationIndices());
            Value back =
                TransferOp::create(builder, loc, TypeRange{value.getType()},
                                   result, place, threadAttr, IntegerAttr())
                    .getResult()
                    .front();
            through.getSourceMutable().assign(tensor::ExpandShapeOp::create(
                builder, loc, expand.getResultType(), back,
                expand.getReassociationIndices()));
            expand->erase();
            continue;
          }
          builder.setInsertionPoint(first);
          // A result inserted into a larger value comes back into its place
          // there (the rows of a part, SplitRows.cpp): lower-hart makes the
          // launch write the destination slice, no copy on the CPU.
          Value target = backs[index];
          if (auto insert = dyn_cast<tensor::InsertSliceOp>(first);
              insert && insert.getSource() == value)
            target = tensor::ExtractSliceOp::create(
                builder, loc, cast<RankedTensorType>(value.getType()),
                insert.getDest(), insert.getMixedOffsets(),
                insert.getMixedSizes(), insert.getMixedStrides());
          Value back =
              TransferOp::create(builder, loc, TypeRange{value.getType()},
                                 result, target, threadAttr, IntegerAttr())
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

    // Subgraphs equal but for their locations (the layers of a model) are one
    // function: the launches of the others call the first.
    SetVector<StringAttr> callees;
    module.walk([&](LaunchOp launch) {
      callees.insert(launch.getCalleeAttr().getAttr());
    });
    llvm::StringMap<StringAttr> unique;
    for (StringAttr name : callees) {
      auto subgraph = module.lookupSymbol<func::FuncOp>(name);
      subgraph.setName("subgraph");
      std::string code;
      llvm::raw_string_ostream printed(code);
      subgraph.print(printed,
                     OpPrintingFlags().useLocalScope().enableDebugInfo(false));
      subgraph.setName(name);
      auto [found, inserted] = unique.try_emplace(code, name);
      if (inserted)
        continue;
      if (failed(
              SymbolTable::replaceAllSymbolUses(name, found->second, module)))
        llvm::report_fatal_error("convert-linalg-to-hart: subgraph uses");
      subgraph.erase();
    }
  }
};

PassRegistration<ConvertLinalgToHartPass> registration;
} // namespace
