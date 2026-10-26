#include "Conversion/LowerLinalgToBuckyball/LowerLinalgToBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

void mlir::buddy::formMegaKernelRegions(ArrayRef<linalg::GenericOp> stageList) {
  SmallVector<linalg::GenericOp> stages(stageList);
  DenseMap<Value, unsigned> producers;
  DenseMap<Operation *, unsigned> positions;
  for (auto [index, op] : llvm::enumerate(stages)) {
    producers[op.getOutputs()[0]] = index;
    positions[op.getOperation()] = index;
  }
  SmallVector<int64_t> next(stages.size(), -1), previous(stages.size(), -1);
  for (auto [index, op] : llvm::enumerate(stages)) {
    Value output = op.getOutputs()[0];
    if (!cast<MemRefType>(output.getType()).getElementType().isInteger(8))
      continue;
    SmallVector<unsigned> consumers;
    bool external = false;
    for (Operation *user : output.getUsers()) {
      if (user == op || isa<memref::DeallocOp>(user))
        continue;
      auto found = positions.find(user);
      if (found == positions.end()) {
        external = true;
        break;
      }
      if (!llvm::is_contained(consumers, found->second))
        consumers.push_back(found->second);
    }
    if (external || consumers.size() != 1)
      continue;
    unsigned target = consumers.front();
    unsigned parents = 0;
    for (Value input : stages[target].getInputs())
      parents += producers.contains(input);
    if (parents != 1 || op->getBlock() != stages[target]->getBlock() ||
        op->hasAttr("buckyball.quantized_matmul") !=
            stages[target]->hasAttr("buckyball.quantized_matmul"))
      continue;
    next[index] = target;
    previous[target] = index;
  }
  unsigned group = 0;
  for (unsigned root = 0; root < stages.size(); ++root) {
    if (previous[root] >= 0)
      continue;
    SmallVector<unsigned> chain;
    for (int64_t index = root; index >= 0; index = next[index])
      chain.push_back(index);
    auto context = stages[root].getContext();
    Builder builder(context);
    auto id = builder.getStringAttr("core." + std::to_string(group++));
    for (auto [index, position] : llvm::enumerate(chain)) {
      auto op = stages[position];
      if (op->hasAttr("core.packed_weight")) {
        Value packed = op.getInputs()[1];
        for (Operation *user : llvm::make_early_inc_range(packed.getUsers()))
          if (isa<memref::DeallocOp>(user))
            user->moveAfter(stages[chain.back()]);
        op->removeAttr("core.packed_weight");
      }

      SmallVector<StringAttr> remove;
      SmallVector<NamedAttribute> add;
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name.consume_front("buckyball.quantized_")) {
          remove.push_back(attr.getName());
          add.emplace_back(builder.getStringAttr("buckyball.mega_" + name),
                           attr.getValue());
        }
      }
      for (StringAttr name : remove)
        op->removeAttr(name);
      for (NamedAttribute attr : add)
        op->setAttr(attr.getName(), attr.getValue());
      op->setAttr("buckyball.mega_kernel", builder.getBoolAttr(true));
      op->setAttr("mega_kernel_id", id);
      op->setAttr("mega_kernel_stage", builder.getI64IntegerAttr(index));
      op->setAttr("mega_kernel_size", builder.getI64IntegerAttr(chain.size()));
    }
  }
}

void mlir::buddy::prepareQuantizedWeights(
    ArrayRef<linalg::GenericOp> stageList) {
  SmallVector<linalg::GenericOp> stages(stageList);
  for (auto op : stages) {
    bool matrix = op->hasAttr("buckyball.quantized_matmul");
    bool conv = op->hasAttr("buckyball.quantized_conv2d");
    bool depthwise = op->hasAttr("buckyball.quantized_conv2d_depthwise");
    if (!matrix && !conv && !depthwise)
      continue;
    Value weight = op.getInputs()[1];
    auto type = cast<MemRefType>(weight.getType());
    auto shape = type.getShape();
    // Pebble's large-convolution lowering reads the logical weights itself.
    if (conv && op->getAttrOfType<IntegerAttr>("kernel").getInt() > 7)
      continue;
    SmallVector<int64_t> packedShape;
    if (matrix)
      packedShape = {shape[1], shape[0]};
    else if (depthwise)
      packedShape = {shape[2], shape[3], shape[0], 1};
    else
      packedShape = {(shape[0] + 15) / 16, shape[1],
                     ((shape[2] * shape[3] + 15) / 16) * 16, 16};
    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value packed = memref::AllocOp::create(
        builder, loc, MemRefType::get(packedShape, type.getElementType()));
    Value zeroByte = arith::ConstantIntOp::create(builder, loc, 0, 8);
    linalg::FillOp::create(builder, loc, zeroByte, packed);
    SmallVector<Value> indices;
    for (int64_t extent : shape) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      Value end = arith::ConstantIndexOp::create(builder, loc, extent);
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      auto loop = scf::ForOp::create(builder, loc, zero, end, one);
      builder.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
    }
    Value value = memref::LoadOp::create(builder, loc, weight, indices);
    SmallVector<Value> destination;
    if (matrix)
      destination = {indices[1], indices[0]};
    else if (depthwise)
      destination = {indices[2], indices[3], indices[0], indices[1]};
    else {
      Value lanes = arith::ConstantIndexOp::create(builder, loc, 16);
      Value width = arith::ConstantIndexOp::create(builder, loc, shape[3]);
      destination = {arith::DivUIOp::create(builder, loc, indices[0], lanes),
                     indices[1],
                     arith::AddIOp::create(
                         builder, loc,
                         arith::MulIOp::create(builder, loc, indices[2], width),
                         indices[3]),
                     arith::RemUIOp::create(builder, loc, indices[0], lanes)};
    }
    memref::StoreOp::create(builder, loc, value, packed, destination);
    op.getInputsMutable()[1].set(packed);
    if (matrix) {
      auto maps = op.getIndexingMapsArray();
      maps[1] = AffineMap::get(3, 0,
                               {getAffineDimExpr(2, op.getContext()),
                                getAffineDimExpr(1, op.getContext())},
                               op.getContext());
      op.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
    }
    builder.setInsertionPointAfter(op);
    memref::DeallocOp::create(builder, loc, packed);
    op->setAttr("core.packed_weight", UnitAttr::get(op.getContext()));
  }
}
