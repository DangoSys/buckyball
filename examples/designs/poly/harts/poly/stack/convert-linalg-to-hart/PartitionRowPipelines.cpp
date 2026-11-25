#include "Acc/AccOps.h"
#include "Hart/HartOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace ::buddy::hart;
using namespace ::buddy::acc;

bool partitionRowPipeline(func::FuncOp function, ArrayRef<int64_t> threads,
                          ArrayRef<StringRef> profiles, int64_t programBank,
                          int64_t bankRows, int64_t linkBanks) {
  Block &body = function.getBody().front();
  if (function.getNumResults() != 1 ||
      llvm::count_if(
          body, [](Operation &op) { return isa<linalg::MatmulOp>(op); }) != 2 ||
      llvm::any_of(body, [](Operation &op) {
        return isa<linalg::BatchMatmulOp, linalg::GenericOp>(op);
      }))
    return false;
  auto result = dyn_cast<RankedTensorType>(function.getResultTypes()[0]);
  if (!result || result.getRank() != 2 || !result.isDynamicDim(0) ||
      result.isDynamicDim(1))
    return false;
  auto returned = cast<func::ReturnOp>(body.getTerminator());
  auto written =
      returned.getOperand(0)
          .getDefiningOp<bufferization::MaterializeInDestinationOp>();
  auto destination =
      written ? dyn_cast<BlockArgument>(written.getDest()) : BlockArgument();
  if (!destination || destination.getOwner() != &body ||
      destination.getArgNumber() + 1 != function.getNumArguments() ||
      destination.getType() != result)
    return false;
  unsigned outputIndex = destination.getArgNumber();
  int64_t inputIndex = -1;
  for (auto [i, argument] : llvm::enumerate(body.getArguments())) {
    if (i == outputIndex)
      continue;
    auto type = dyn_cast<RankedTensorType>(argument.getType());
    if (!type || (type.getRank() == 0 && !argument.use_empty()))
      return false;
    if (!type.hasStaticShape()) {
      if (inputIndex >= 0 || type.getRank() != 2 || !type.isDynamicDim(0) ||
          type.isDynamicDim(1) || !type.getElementType().isInteger(8))
        return false;
      inputIndex = i;
    }
  }
  if (inputIndex < 0)
    return false;
  SmallVector<unsigned> weights;
  int64_t weightBanks = 0;
  for (auto argument : body.getArguments()) {
    auto type = cast<RankedTensorType>(argument.getType());
    if (!type.hasStaticShape() || type.getRank() == 0 ||
        !type.getElementType().isInteger(8) || type.getNumElements() % 16)
      continue;
    weights.push_back(argument.getArgNumber());
    weightBanks +=
        (type.getNumElements() + bankRows * 16 - 1) / (bankRows * 16);
  }
  if (weightBanks > linkBanks || weightBanks < 2)
    weights.clear();
  auto module = function->getParentOfType<ModuleOp>();
  auto loc = function.getLoc();
  OpBuilder b(function);
  function.setArgAttr(outputIndex, "bufferization.writable",
                      b.getBoolAttr(true));
  SmallVector<func::FuncOp> kernels;
  for (size_t part = 0; part < threads.size(); ++part) {
    auto kernel = cast<func::FuncOp>(function->clone());
    kernel.setName(
        (function.getName() + "_pipeline" + llvm::Twine(part)).str());
    module.push_back(kernel);
    kernel.setPrivate();
    kernel->setAttr("bbacc.target", b.getStringAttr(profiles[part]));
    kernel->setAttr("hart.program_bank", b.getI64IntegerAttr(programBank));
    kernel->setAttr("hart.parallel_group",
                    b.getStringAttr((function.getName() + "_pipeline").str()));
    kernel->setAttr("hart.parallel_part", b.getI64IntegerAttr(part));
    kernel.setArgAttr(
        kernel.getNumArguments() - 1, "bufferization.buffer_layout",
        AffineMapAttr::get(b.getMultiDimIdentityMap(result.getRank())));
    kernel.setArgAttr(kernel.getNumArguments() - 1, "bufferization.writable",
                      b.getBoolAttr(true));
    b.setInsertionPointToStart(&kernel.getBody().front());
    Value rows =
        tensor::DimOp::create(b, loc, kernel.getArgument(inputIndex), 0);
    SmallVector<Operation *> computations;
    for (Operation &op : kernel.getBody().front())
      if (isa<linalg::MatmulOp>(op) ||
          op.getName().getStringRef() == "frontend.mxint8_activation" ||
          (isa<::buddy::acc::CustomOp>(op) &&
           cast<::buddy::acc::CustomOp>(op).getKind() == "row_operator"))
        computations.push_back(&op);
    for (Operation *op : computations) {
      Value value = op->getResult(0);
      if (value.hasOneUse() && isa<bufferization::MaterializeInDestinationOp>(
                                   *value.getUsers().begin()))
        continue;
      auto type = cast<RankedTensorType>(value.getType());
      b.setInsertionPoint(op);
      SmallVector<Value> dynamic(type.getNumDynamicDims(), rows);
      Value target = tensor::EmptyOp::create(b, loc, type.getShape(),
                                             type.getElementType(), dynamic);
      b.setInsertionPointAfter(op);
      auto materialized = bufferization::MaterializeInDestinationOp::create(
          b, loc, value, target);
      value.replaceAllUsesExcept(materialized.getResult(), materialized);
    }
    kernels.push_back(kernel);
  }
  function.getBody().dropAllReferences();
  function.getBody().getBlocks().clear();
  Block *host = function.addEntryBlock();
  b.setInsertionPointToStart(host);
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  SmallVector<Value> inputs;
  for (Value argument : host->getArguments())
    inputs.push_back(argument);
  if (!weights.empty()) {
    SmallVector<Type> types;
    SmallVector<Value> sources, targets;
    for (unsigned argument : weights) {
      auto type = cast<RankedTensorType>(inputs[argument].getType());
      types.push_back(type);
      Value staged = tensor::EmptyOp::create(b, loc, type.getShape(),
                                             type.getElementType());
      sources.push_back(TransferOp::create(
                            b, loc, TypeRange{type}, inputs[argument], staged,
                            IntegerAttr(), b.getI64IntegerAttr(threads.front()))
                            .getResult()
                            .front());
      targets.push_back(tensor::EmptyOp::create(b, loc, type.getShape(),
                                                type.getElementType()));
    }
    auto name = (function.getName() + "_weights").str();
    auto load = LaunchOp::create(
        b, loc, types, FlatSymbolRefAttr::get(b.getContext(), name), sources,
        targets, b.getI64IntegerAttr(threads.front()));
    for (auto [i, argument] : llvm::enumerate(weights))
      inputs[argument] = load.getResult(i);
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToEnd(module.getBody());
    SmallVector<Type> arguments(types);
    arguments.append(types);
    auto prefetch =
        func::FuncOp::create(b, loc, name, b.getFunctionType(arguments, types));
    prefetch.setPrivate();
    prefetch->setAttr("bbacc.target", b.getStringAttr(profiles.front()));
    prefetch->setAttr("hart.program_bank", b.getI64IntegerAttr(programBank));
    Block *entry = prefetch.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    SmallVector<Value> outputs;
    for (auto [i, tensorType] : llvm::enumerate(types)) {
      auto type = cast<RankedTensorType>(tensorType);
      for (unsigned argument : {unsigned(i), unsigned(types.size() + i)})
        prefetch.setArgAttr(
            argument, "bufferization.buffer_layout",
            AffineMapAttr::get(b.getMultiDimIdentityMap(type.getRank())));
      prefetch.setArgAttr(types.size() + i, "bufferization.writable",
                          b.getBoolAttr(true));
      auto memory = [&](Value value) -> Value {
        if (type.getRank() > 1) {
          ReassociationIndices axes;
          for (int64_t axis = 0; axis < type.getRank(); ++axis)
            axes.push_back(axis);
          value = memref::CollapseShapeOp::create(
              b, loc, value, ArrayRef<ReassociationIndices>{axes});
        }
        return value;
      };
      auto memoryType = MemRefType::get(type.getShape(), type.getElementType());
      Value source = memory(bufferization::ToBufferOp::create(
          b, loc, memoryType, entry->getArgument(i)));
      Value output = bufferization::ToBufferOp::create(
          b, loc, memoryType, entry->getArgument(types.size() + i));
      Value target = memory(output);
      auto copy = [&](Value first, int64_t rows) {
        auto slice = [&](Value value) -> Value {
          return memref::SubViewOp::create(
              b, loc, value, ArrayRef<OpFoldResult>{first},
              ArrayRef<OpFoldResult>{b.getIndexAttr(rows * 16)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        };
        Value loaded = BankMvinOp::create(b, loc, b.getI64Type(), slice(source),
                                          bank, rows, 1);
        BankMvoutOp::create(b, loc, b.getI64Type(), slice(target), loaded, rows,
                            1);
      };
      int64_t rows = type.getNumElements() / 16;
      if (rows / bankRows) {
        auto loop = scf::ForOp::create(b, loc, index(0), index(rows / bankRows),
                                       index(1));
        loop->setAttr("hart.stream", b.getUnitAttr());
        OpBuilder::InsertionGuard bodyInsertion(b);
        b.setInsertionPointToStart(loop.getBody());
        copy(arith::MulIOp::create(b, loc, loop.getInductionVar(),
                                   index(bankRows * 16)),
             bankRows);
      }
      if (rows % bankRows)
        copy(index(rows / bankRows * bankRows * 16), rows % bankRows);
      outputs.push_back(
          bufferization::ToTensorOp::create(b, loc, type, output, true, true));
    }
    BankReleaseOp::create(b, loc, bank);
    func::ReturnOp::create(b, loc, outputs);
  }
  Value source = host->getArgument(inputIndex);
  auto input = cast<RankedTensorType>(source.getType());
  Value rows = tensor::DimOp::create(b, loc, source, 0);
  SmallVector<Value> starts, lengths, launches;
  for (auto [part, thread] : llvm::enumerate(threads)) {
    auto boundary = [&](int64_t n) -> Value {
      return arith::DivUIOp::create(
          b, loc, arith::MulIOp::create(b, loc, rows, index(n)),
          index(threads.size()));
    };
    Value start = boundary(part),
          length = arith::SubIOp::create(b, loc, boundary(part + 1), start);
    starts.push_back(start);
    lengths.push_back(length);
    Value piece = tensor::ExtractSliceOp::create(
        b, loc, source, ArrayRef<OpFoldResult>{start, b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{length, b.getIndexAttr(input.getDimSize(1))},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    auto threadAttr = b.getI64IntegerAttr(thread);
    SmallVector<Value> operands;
    for (auto [i, argument] : llvm::enumerate(host->getArguments())) {
      if (i == outputIndex)
        continue;
      Value value = int64_t(i) == inputIndex ? piece : inputs[i];
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<Value> dynamic;
      if (!type.hasStaticShape())
        dynamic.push_back(length);
      Value target = tensor::EmptyOp::create(b, loc, type.getShape(),
                                             type.getElementType(), dynamic);
      IntegerAttr from = llvm::is_contained(weights, i)
                             ? b.getI64IntegerAttr(threads.front())
                             : IntegerAttr();
      operands.push_back(TransferOp::create(b, loc, TypeRange{type}, value,
                                            target, from, threadAttr)
                             .getResult()
                             .front());
    }
    Value output = tensor::EmptyOp::create(
        b, loc, result.getShape(), result.getElementType(), ValueRange{length});
    // The launch computes the kernel's result here: its location.
    Location computed = kernels[part]
                            .front()
                            .getTerminator()
                            ->getOperand(0)
                            .getDefiningOp()
                            ->getLoc();
    launches.push_back(
        LaunchOp::create(b, computed, TypeRange{result},
                         FlatSymbolRefAttr::get(kernels[part].getNameAttr()),
                         operands, ValueRange{output}, threadAttr)
            .getResult(0));
  }
  Value output = host->getArgument(outputIndex);
  for (auto [part, thread] : llvm::enumerate(threads)) {
    Value target = tensor::ExtractSliceOp::create(
        b, loc, output, ArrayRef<OpFoldResult>{starts[part], b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{lengths[part],
                               b.getIndexAttr(result.getDimSize(1))},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value value =
        TransferOp::create(b, loc, TypeRange{result}, launches[part], target,
                           b.getI64IntegerAttr(thread), IntegerAttr())
            .getResult()
            .front();
    output = tensor::InsertSliceOp::create(
        b, loc, value, output,
        ArrayRef<OpFoldResult>{starts[part], b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{lengths[part],
                               b.getIndexAttr(result.getDimSize(1))},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  }
  func::ReturnOp::create(b, loc, output);
  return true;
}
