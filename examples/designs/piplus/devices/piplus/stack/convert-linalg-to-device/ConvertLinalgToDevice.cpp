//===- ConvertLinalgToDevice.cpp - A program on the piplus harts ----------===//
//
// convert-linalg-to-device chooses the hart of every launch of the program,
// which runs on the control hart.
//
// A launch runs on the hart of its class that already holds the most bytes
// of its inputs, the values it computed or that were transferred to it: the
// denoising steps stay on the hart of the first, which received the
// prefix's keys and values, and the prefix on a hart that encoded a camera.
// A launch whose inputs no hart of its class holds goes to the next hart of
// the class in the order of design.pb: the vision encoders of the cameras
// spread over the dense harts. A launch of a rank of a partitioned graph
// (`device.rank`) runs on the hart of its rank: the ranks of a class, in
// their order, on the harts of the class in the order of design.pb, the
// reduce hart's with the first rank of its class.
//
// The harts share the DDR, coherent in hardware (the device Home answers a
// thread's DMA for every cache): a launch reads a value of another hart in
// place, once the program has run the launch that wrote it (lower-device
// orders them). An operation of the control hart that reads a value of a
// launch reads a transfer of it to the control hart (lower-device waits for
// the launch there), made once per value, before its first reader. A value
// lives on the hart that computed it: the results of a launch on its hart,
// the operations between the launches on the control hart, what is in the
// DDR when the program starts (its arguments, constants and their slices)
// on none; a slice of a launch's result that only launches read on the hart
// of the launch. The program is marked with its hart, `device.hart`.
//
//===----------------------------------------------------------------------===//

#include "Device/DeviceOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MemoryBuffer.h"

#include <limits>
#include <set>

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
    return "Place the launches of the program on the piplus harts and "
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
    // The harts of each class, by their index in the device.
    llvm::StringMap<SmallVector<int64_t>> classes;
    for (auto [index, hart] : llvm::enumerate(design.device().harts()))
      classes[hart.component()].push_back(index);
    int64_t control = classes.lookup("control").front();

    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || function.getOps<LaunchOp>().empty())
        continue;
      // The ranks of each class, in their order.
      llvm::StringMap<std::set<int64_t>> ranks;
      for (LaunchOp launch : function.getOps<LaunchOp>())
        if (auto rank = launch->getAttrOfType<IntegerAttr>("device.rank"))
          ranks[*launch.getHartClass()].insert(rank.getInt());
      // The program runs on the control hart.
      function->setAttr(
          "device.hart",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), control));
      // The values in the DDR when the program starts.
      DenseSet<Value> memory(function.getArguments().begin(),
                             function.getArguments().end());
      llvm::StringMap<size_t> turns;
      DenseMap<Value, int64_t> homes;
      auto home = [&](Value value) {
        auto found = homes.find(value);
        return found == homes.end() ? control : found->second;
      };
      DenseMap<Value, Value> copies;
      // The operand an operation of the control hart reads, transferred
      // there before its first reader `user`.
      auto fetch = [&](OpOperand &operand, Operation *user) {
        Value value = operand.get();
        int64_t from = home(value);
        if (memory.contains(value) || from == control)
          return;
        Value &copy = copies[value];
        if (!copy) {
          OpBuilder builder(user);
          auto type = cast<RankedTensorType>(value.getType());
          Value target = tensor::EmptyOp::create(
              builder, user->getLoc(), type.getShape(), type.getElementType());
          copy = TransferOp::create(builder, user->getLoc(), TypeRange{type},
                                    value, target, from, control)
                     .getResult()
                     .front();
          homes[copy] = control;
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
        // A slice of a launch's result that only launches read (a rank's
        // slice of the prefix's keys and values) is a view of it on its hart:
        // the launches read it in place.
        if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
            slice && homes.contains(slice.getSource()) &&
            llvm::all_of(slice->getUsers(),
                         [](Operation *user) { return isa<LaunchOp>(user); })) {
          homes[slice.getResult()] = homes.lookup(slice.getSource());
          continue;
        }
        auto launch = dyn_cast<LaunchOp>(op);
        if (!launch) {
          for (OpOperand &operand : op.getOpOperands())
            fetch(operand, &op);
          continue;
        }
        const SmallVector<int64_t> &harts = classes.at(*launch.getHartClass());
        int64_t hart = harts[turns[*launch.getHartClass()]++ % harts.size()];
        int64_t most = 0;
        auto rank = launch->getAttrOfType<IntegerAttr>("device.rank");
        if (rank) {
          const std::set<int64_t> &own = ranks.at(*launch.getHartClass());
          if (own.size() > harts.size())
            llvm::report_fatal_error("convert-linalg-to-device: more ranks "
                                     "than harts of their class");
          hart = harts[std::distance(own.begin(), own.find(rank.getInt()))];
          most = std::numeric_limits<int64_t>::max();
        }
        for (int64_t candidate : harts) {
          int64_t bytes = 0;
          for (Value input : launch.getInputs()) {
            auto type = cast<RankedTensorType>(input.getType());
            if (!memory.contains(input) && home(input) == candidate)
              bytes += type.getNumElements() *
                       ((type.getElementTypeBitWidth() + 7) / 8);
          }
          if (bytes > most) {
            most = bytes;
            hart = candidate;
          }
        }
        launch.removeHartClassAttr();
        launch.setHart(hart);
        for (Value result : launch.getResults())
          homes[result] = hart;
      }
    }
  }
};

PassRegistration<ConvertLinalgToDevicePass> registration;
} // namespace
