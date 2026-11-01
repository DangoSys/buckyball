//===- ConvertLinalgToDevice.cpp - A program on the poly cores ------------===//
//
// convert-linalg-to-device chooses the core of every launch of a device program
// and inserts the transfers between the cores. The program runs on the core of
// the control hart (design.pb), the reduce core.
//
// A launch of a class goes to the core of the class that already holds the
// most bytes of its inputs; one that reads no value of a core of its class
// takes the cores of the class in turn. The ranks of a tensor-parallel stage
// are launched in rank order, so each stage deals them out over the poly
// cores the same way.
//
// The arguments of the program are in the DDR every core reads: weights,
// inputs and states, and the slices the program takes of them. They do not
// move. Every other value lives on the core that computed it: a launch's
// results on its core, the program's own operations on the program's core.
// A launch that reads a value of another core, or an operation of the
// program that reads a value of a launch, reads a transfer of it to its own
// core, made once per value and core before its first reader. The
// destinations a launch writes are not read, so they do not move.
//
// Values move over the poly core links (arch/device/CoreLinks.scala): they
// route between the reduce core (core 0) and every poly core, both ways, and
// not between two poly cores. A value between two poly cores goes through
// core 0. The program is marked with its
// core, `device.core`.
//
//===----------------------------------------------------------------------===//

#include "Device/DeviceOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::device;

namespace {
class ConvertLinalgToDevicePass
    : public PassWrapper<ConvertLinalgToDevicePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToDevicePass)

  ConvertLinalgToDevicePass() = default;
  ConvertLinalgToDevicePass(const ConvertLinalgToDevicePass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "convert-linalg-to-device"; }
  StringRef getDescription() const final {
    return "Place the launches of the program on the poly cores and "
           "transfer the values between them over the core links";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<DeviceDialect, tensor::TensorDialect>();
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
    // The cores of each class, by their index in the device, and the core
    // of the control hart.
    llvm::StringMap<SmallVector<int64_t>> classes;
    int64_t control = -1;
    for (auto [index, core] : llvm::enumerate(design.device().cores())) {
      classes[core.component()].push_back(index);
      if (core.hart_id() == design.workload().control_hart())
        control = index;
    }
    if (control < 0)
      llvm::report_fatal_error("no core runs the control hart");
    // The routes of the core links: core 0 and every other core.
    auto linked = [](int64_t a, int64_t b) { return a == 0 || b == 0; };

    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || function.getOps<LaunchOp>().empty())
        continue;
      function->setAttr(
          "device.core",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), control));
      // The values in the DDR: the arguments and the slices of them.
      DenseSet<Value> memory(function.getArguments().begin(),
                             function.getArguments().end());
      llvm::StringMap<size_t> turns;
      DenseMap<Value, int64_t> homes;
      DenseMap<std::pair<Value, int64_t>, Value> copies;
      auto home = [&](Value value) {
        auto found = homes.find(value);
        return found == homes.end() ? control : found->second;
      };
      // `value` on `core`, transferred there before `user` if it is not.
      std::function<Value(Value, int64_t, Operation *)> fetch =
          [&](Value value, int64_t core, Operation *user) -> Value {
        int64_t from = home(value);
        if (memory.contains(value) || from == core)
          return value;
        Value &copy = copies[{value, core}];
        if (copy)
          return copy;
        Value source = linked(from, core) ? value : fetch(value, 0, user);
        OpBuilder builder(user);
        auto type = cast<RankedTensorType>(value.getType());
        SmallVector<Value> sizes;
        for (auto [axis, size] : llvm::enumerate(type.getShape()))
          if (ShapedType::isDynamic(size))
            sizes.push_back(
                tensor::DimOp::create(builder, user->getLoc(), source, axis));
        Value target =
            tensor::EmptyOp::create(builder, user->getLoc(), type.getShape(),
                                    type.getElementType(), sizes);
        Value moved =
            TransferOp::create(builder, user->getLoc(), TypeRange{type}, source,
                               target, home(source), core)
                .getResult()
                .front();
        homes[moved] = core;
        copies[{value, core}] = moved;
        return moved;
      };
      // The bytes of a value; a dynamic size counts once.
      auto bytes = [](Value value) {
        auto type = cast<ShapedType>(value.getType());
        int64_t elements = 1;
        for (int64_t size : type.getShape())
          elements *= ShapedType::isDynamic(size) ? 1 : size;
        return elements * ((type.getElementTypeBitWidth() + 7) / 8);
      };
      for (Operation &op : llvm::make_early_inc_range(function.getOps())) {
        if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
            slice && memory.contains(slice.getSource())) {
          memory.insert(slice.getResult());
          continue;
        }
        // The sizes of a value are known on every core.
        if (isa<tensor::DimOp>(op))
          continue;
        auto launch = dyn_cast<LaunchOp>(op);
        if (!launch) {
          for (OpOperand &operand : op.getOpOperands())
            if (isa<RankedTensorType>(operand.get().getType()))
              operand.set(fetch(operand.get(), control, &op));
          continue;
        }
        StringRef name = *launch.getCoreClass();
        const SmallVector<int64_t> &cores = classes.at(name);
        DenseMap<int64_t, int64_t> held;
        for (Value input : launch.getInputs())
          if (!memory.contains(input) && homes.contains(input))
            held[homes[input]] += bytes(input);
        int64_t core = -1;
        int64_t most = 0;
        for (int64_t candidate : cores)
          if (held.lookup(candidate) > most) {
            most = held.lookup(candidate);
            core = candidate;
          }
        if (core < 0)
          core = cores[turns[name]++ % cores.size()];
        launch.removeCoreClassAttr();
        launch.setCore(core);
        for (OpOperand &operand : launch.getInputsMutable())
          operand.set(fetch(operand.get(), core, launch));
        for (Value result : launch.getResults())
          homes[result] = core;
      }
    }
  }
};

PassRegistration<ConvertLinalgToDevicePass> registration;
} // namespace
