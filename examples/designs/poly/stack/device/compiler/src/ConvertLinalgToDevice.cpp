//===- ConvertLinalgToDevice.cpp - A program on the poly cores ------------===//
//
// convert-linalg-to-device chooses the core of every launch of a device program
// sharing coherent DDR. The program runs on the core of
// the control hart (design.pb), the reduce core.
//
// Each launch names its logical rank within a core class. The design maps
// that rank to the corresponding configured core, preserving parallel work
// even when all ranks read one shared producer.
//
// Every core accesses the same coherent DDR. Results remain in their
// producer's allocation and consumers read that buffer after its producer
// completes. LowerDevice records those job dependencies and keeps temporary
// buffers alive until all consumers finish; no per-core DDR copy is needed.
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
    return "Place the launches on poly cores sharing coherent DDR";
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

    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || function.getOps<LaunchOp>().empty())
        continue;
      function->setAttr(
          "device.core",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), control));
      for (auto launch : function.getOps<LaunchOp>()) {
        const auto &cores = classes.at(*launch.getCoreClass());
        auto rank = launch->getAttrOfType<IntegerAttr>("device.rank");
        if (!rank || rank.getInt() < 0 ||
            uint64_t(rank.getInt()) >= cores.size()) {
          launch.emitError(
              "device.rank must name a configured core of its class");
          return signalPassFailure();
        }
        launch.setCore(cores[rank.getInt()]);
        launch.removeCoreClassAttr();
        launch->removeAttr("device.rank");
      }
    }
  }
};

PassRegistration<ConvertLinalgToDevicePass> registration;
} // namespace
