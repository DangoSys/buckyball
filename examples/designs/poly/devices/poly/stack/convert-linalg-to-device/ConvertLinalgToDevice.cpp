//===- ConvertLinalgToDevice.cpp - A program on the poly harts ------------===//
//
// convert-linalg-to-device places every launch of a device program on a hart
// and inserts the transfers between the harts. The program runs on the hart
// of the control hart (design.pb), the reduce hart.
//
// Each launch names its rank within its hart class; rank r runs on the r-th
// configured hart of the class.
//
// The arguments and the results of the program are in the DDR every hart
// reads: weights, inputs and states, and the slices the program takes of
// them, do not move, and a launch writes its slice of a result in place. Every
// other value lives on the hart that computed it: a launch's results on its
// hart, the program's own operations on the program's hart. A launch that
// reads a value of another hart, or an operation of the program that reads a
// value of a launch, reads a transfer of it to its own hart, made once per
// value and hart before its first reader.
//
// Values move through the device SHM over the hart links
// (arch/device/HartLinks.scala): the routers of the device SHM network route
// a write between any two harts over the kept links of the mesh, so a value
// moves to a hart in one transfer from the hart it is on; its producer pushes
// it, its first reader there pulls it (stack/device/runtime/device.c). No
// launch on the receiving hart may block that push while waiting for its
// sender. Such transfers are dispatched at their reader's program position.
// The program is marked with its hart, `device.hart`.
//
//===----------------------------------------------------------------------===//

#include "Device/DeviceOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include <functional>

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
    return "Place the launches of the program on the poly harts and "
           "transfer the values between them through the device SHM";
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
    // The harts of each class, by their index in the device, and the hart
    // of the control hart.
    llvm::StringMap<SmallVector<int64_t>> classes;
    int64_t control = -1;
    for (auto [index, hart] : llvm::enumerate(design.device().harts())) {
      classes[hart.component()].push_back(index);
      if (hart.hart_id() == design.workload().control_hart())
        control = index;
    }
    if (control < 0)
      llvm::report_fatal_error("no hart runs the control hart");
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || function.getOps<LaunchOp>().empty())
        continue;
      function->setAttr(
          "device.hart",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), control));
      // The values in the DDR: the arguments and the slices of them.
      DenseSet<Value> memory(function.getArguments().begin(),
                             function.getArguments().end());
      DenseMap<Value, int64_t> homes;
      DenseMap<std::pair<Value, int64_t>, Value> copies;
      auto home = [&](Value value) {
        auto found = homes.find(value);
        return found == homes.end() ? control : found->second;
      };
      // `value` on `hart`, transferred there before `user` if it is not.
      auto fetch = [&](Value value, int64_t hart, Operation *user) -> Value {
        int64_t from = home(value);
        if (memory.contains(value) || from == hart)
          return value;
        Value &copy = copies[{value, hart}];
        if (copy)
          return copy;
        OpBuilder builder(user);
        auto type = cast<RankedTensorType>(value.getType());
        SmallVector<Value> sizes;
        for (auto [axis, size] : llvm::enumerate(type.getShape()))
          if (ShapedType::isDynamic(size))
            sizes.push_back(
                tensor::DimOp::create(builder, user->getLoc(), value, axis));
        Value target =
            tensor::EmptyOp::create(builder, user->getLoc(), type.getShape(),
                                    type.getElementType(), sizes);
        Value moved =
            TransferOp::create(builder, user->getLoc(), TypeRange{type}, value,
                               target, from, hart)
                .getResult()
                .front();
        homes[moved] = hart;
        copies[{value, hart}] = moved;
        return moved;
      };
      for (Operation &op : llvm::make_early_inc_range(function.getOps())) {
        if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
            slice && memory.contains(slice.getSource())) {
          memory.insert(slice.getResult());
          continue;
        }
        // The sizes of a value are known on every hart; the results of the
        // program are written in place.
        if (isa<tensor::DimOp, func::ReturnOp>(op))
          continue;
        // A launch that wrote its slice of a result in place.
        if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) {
          auto result = dyn_cast<OpResult>(insert.getSource());
          auto launch =
              result ? dyn_cast<LaunchOp>(result.getOwner()) : nullptr;
          auto slice = launch ? launch.getOuts()[result.getResultNumber()]
                                    .getDefiningOp<tensor::ExtractSliceOp>()
                              : nullptr;
          if (slice && slice.getSource() == insert.getDest() &&
              slice.getMixedOffsets() == insert.getMixedOffsets() &&
              slice.getMixedSizes() == insert.getMixedSizes() &&
              slice.getMixedStrides() == insert.getMixedStrides())
            continue;
        }
        auto launch = dyn_cast<LaunchOp>(op);
        if (!launch) {
          for (OpOperand &operand : op.getOpOperands())
            if (isa<RankedTensorType>(operand.get().getType()))
              operand.set(fetch(operand.get(), control, &op));
          continue;
        }
        const auto &harts = classes.at(*launch.getHartClass());
        auto rank = launch->getAttrOfType<IntegerAttr>("device.rank");
        if (!rank || rank.getInt() < 0 ||
            uint64_t(rank.getInt()) >= harts.size()) {
          launch.emitError(
              "device.rank must name a configured hart of its class");
          return signalPassFailure();
        }
        int64_t hart = harts[rank.getInt()];
        launch.setHart(hart);
        launch.removeHartClassAttr();
        launch->removeAttr("device.rank");
        for (OpOperand &operand : launch.getInputsMutable())
          operand.set(fetch(operand.get(), hart, launch));
        for (Value result : launch.getResults())
          homes[result] = hart;
      }
      // The device runtime reads the values of a link in the order they are
      // sent, and a sender may wait for a reader. A launch on the receiving
      // hart between the producer of a value and its first reader there runs
      // first. If it waits for the producer or a later launch of that hart,
      // dispatch this transfer after that earlier receiver work instead.
      SmallVector<LaunchOp> launches(function.getOps<LaunchOp>());
      DenseMap<Operation *, size_t> order;
      for (auto [index, launch] : llvm::enumerate(launches))
        order[launch] = index;
      // The launches each launch waits for, directly or not.
      SmallVector<llvm::BitVector> waits(launches.size(),
                                         llvm::BitVector(launches.size()));
      SmallVector<llvm::BitVector> providers = waits;
      DenseMap<Value, llvm::BitVector> reached;
      std::function<llvm::BitVector(Value)> waitsOf = [&](Value value) {
        auto known = reached.find(value);
        if (known != reached.end())
          return known->second;
        llvm::BitVector found(launches.size());
        if (Operation *op = value.getDefiningOp()) {
          if (auto launch = dyn_cast<LaunchOp>(op)) {
            found.set(order[launch]);
          } else {
            for (Value operand : op->getOperands())
              found |= waitsOf(operand);
          }
        }
        reached[value] = found;
        return found;
      };
      for (auto [index, launch] : llvm::enumerate(launches)) {
        for (Value operand : launch->getOperands())
          providers[index] |= waitsOf(operand);
        waits[index] = providers[index];
        for (size_t provider : providers[index].set_bits())
          waits[index] |= waits[provider];
      }
      for (auto moved : function.getOps<TransferOp>()) {
        auto producer = moved.getSource().getDefiningOp<LaunchOp>();
        if (!producer)
          continue;
        LaunchOp reader;
        for (Operation *user : moved.getResult().front().getUsers())
          if (auto launch = dyn_cast<LaunchOp>(user))
            if (!reader || order[launch] < order[reader])
              reader = launch;
        if (!reader)
          continue;
        const int64_t sender = *producer.getHart();
        // Fan-in must not hold its producers inside blocking pushes. Queue
        // its sends at the reader, after every producer has been dispatched.
        if (providers[order[reader]].count() > 1)
          moved->setAttr("device.defer", UnitAttr::get(&getContext()));
        for (size_t index = order[producer] + 1; index < order[reader];
             ++index) {
          if (*launches[index].getHart() != moved.getTo())
            continue;
          for (size_t waited : waits[index].set_bits())
            if (waited >= order[producer] &&
                *launches[waited].getHart() == sender) {
              // Send at the transfer's program position, after the earlier
              // receiver jobs have been dispatched, instead of blocking the
              // original producer behind a reader that cannot run yet.
              moved->setAttr("device.defer", UnitAttr::get(&getContext()));
            }
        }
      }
      // Keep safe fan-out together: one runtime job stages each chunk once
      // for all receivers. Every receiver's earlier work must precede the
      // first transfer, so that it can drain the link while this job sends.
      DenseMap<Value, SmallVector<TransferOp>> fanouts;
      for (auto moved : function.getOps<TransferOp>())
        if (moved->hasAttr("device.defer"))
          fanouts[moved.getSource()].push_back(moved);
      for (auto &[source, moves] : fanouts) {
        if (moves.size() < 2)
          continue;
        Operation *anchor = moves.front();
        bool ready = true;
        for (TransferOp moved : moves) {
          LaunchOp reader;
          for (Operation *user : moved.getResult().front().getUsers())
            if (auto launch = dyn_cast<LaunchOp>(user))
              if (!reader || order[launch] < order[reader])
                reader = launch;
          if (!reader) {
            ready = false;
            break;
          }
          for (LaunchOp launch : launches)
            if (*launch.getHart() == moved.getTo() &&
                anchor->isBeforeInBlock(launch) &&
                launch->isBeforeInBlock(reader))
              ready = false;
        }
        if (!ready)
          continue;
        std::function<void(Value)> hoist = [&](Value value) {
          Operation *op = value.getDefiningOp();
          if (!op || op->isBeforeInBlock(anchor))
            return;
          if (!isMemoryEffectFree(op))
            llvm::report_fatal_error("a fan-out target has an impure size");
          for (Value operand : op->getOperands())
            hoist(operand);
          op->moveBefore(anchor);
        };
        for (TransferOp moved : moves) {
          hoist(moved.getTarget());
          moved->setAttr("device.fanout", UnitAttr::get(&getContext()));
        }
        Operation *previous = anchor;
        for (TransferOp moved : ArrayRef<TransferOp>(moves).drop_front()) {
          moved->moveAfter(previous);
          previous = moved;
        }
      }
    }
  }
};

PassRegistration<ConvertLinalgToDevicePass> registration;
} // namespace
