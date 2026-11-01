//===- ConvertLinalgToDevice.cpp - A program on the piplus cores ----------===//
//
// convert-linalg-to-device chooses the core of every launch of the program,
// which runs on the control core, and inserts the transfers between the cores.
//
// The cores of a class, in the order of design.pb, take the launches of the
// class in turn: with two dense cores, the vision encoders of the cameras
// alternate between them and the prefix follows on the next one; the
// denoising steps all run on the one action core.
//
// What is in the DDR when the program starts does not move: its arguments
// (weights, inputs, states), its constants and the slices it takes of them.
// Every core reads them in place; how a core loads them is its own business.
// Every other value lives on the core that computed it: the results of a
// launch on its core, the operations between the launches on the control
// core. A launch that reads a value of another core, or an operation of the
// control core that reads a value of a launch, reads a transfer of it to its
// own core; the transfer is made once per value and core, before its first
// reader. The destinations a launch writes are not read, so they do not
// move. The program is marked with its core, `device.core`.
//
//===----------------------------------------------------------------------===//

#include "Device/DeviceOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
    return "Place the launches of the program on the piplus cores and "
           "transfer the values between them";
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
    // The cores of each class, by their index in the device.
    llvm::StringMap<SmallVector<int64_t>> classes;
    for (auto [index, core] : llvm::enumerate(design.device().cores()))
      classes[core.component()].push_back(index);
    int64_t control = classes.lookup("control").front();

    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || function.getOps<LaunchOp>().empty())
        continue;
      // The program runs on the control core.
      function->setAttr(
          "device.core",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), control));
      // The values in the DDR when the program starts.
      DenseSet<Value> memory(function.getArguments().begin(),
                             function.getArguments().end());
      llvm::StringMap<size_t> turns;
      DenseMap<Value, int64_t> homes;
      DenseMap<std::pair<Value, int64_t>, Value> copies;
      auto home = [&](Value value) {
        auto found = homes.find(value);
        return found == homes.end() ? control : found->second;
      };
      // The operand read on `core`, transferred there before its first
      // reader `user`.
      auto fetch = [&](OpOperand &operand, int64_t core, Operation *user) {
        Value value = operand.get();
        int64_t from = home(value);
        if (memory.contains(value) || from == core)
          return;
        Value &copy = copies[{value, core}];
        if (!copy) {
          OpBuilder builder(user);
          auto type = cast<RankedTensorType>(value.getType());
          Value target = tensor::EmptyOp::create(
              builder, user->getLoc(), type.getShape(), type.getElementType());
          copy = TransferOp::create(builder, user->getLoc(), TypeRange{type},
                                    value, target, from, core)
                     .getResult()
                     .front();
          homes[copy] = core;
        }
        operand.set(copy);
      };
      for (Operation &op : llvm::make_early_inc_range(function.getOps())) {
        if (isa<arith::ConstantOp>(op)) {
          memory.insert(op.getResult(0));
          continue;
        }
        if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
            slice && memory.contains(slice.getSource())) {
          memory.insert(slice.getResult());
          continue;
        }
        auto launch = dyn_cast<LaunchOp>(op);
        if (!launch) {
          for (OpOperand &operand : op.getOpOperands())
            fetch(operand, control, &op);
          continue;
        }
        const SmallVector<int64_t> &cores = classes.at(*launch.getCoreClass());
        int64_t core = cores[turns[*launch.getCoreClass()]++ % cores.size()];
        launch.removeCoreClassAttr();
        launch.setCore(core);
        for (OpOperand &operand : launch.getInputsMutable())
          fetch(operand, core, launch);
        for (Value result : launch.getResults())
          homes[result] = core;
      }
    }
  }
};

PassRegistration<ConvertLinalgToDevicePass> registration;
} // namespace
