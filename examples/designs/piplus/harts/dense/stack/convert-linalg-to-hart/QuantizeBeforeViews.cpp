//===- QuantizeBeforeViews.cpp - Quantize a value, not its views ----------===//
//
// Part of convert-linalg-to-hart: a fp32_to_mxint8 of transposes and reshapes
// of a value x, blocked along an axis that is x's last, quantizes x instead
// and gives its decoding the same views. The blocks are the same: an expand
// may split x's last axis only into a last axis of whole 32-value blocks. The
// views then sit between the decoding and the product, where they go with
// the product to the thread, whose DMA reads the blocks of x in place; the
// CPU no longer copies x into the shape of the views (the transposes of k
// into the heads of an attention). x is not a product's result: the product
// writes it from its own subgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

void quantizeBeforeViews(ModuleOp module) {
  SmallVector<Operation *> quantizations;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "frontend.fp32_to_mxint8")
      quantizations.push_back(op);
  });
  for (Operation *quant : quantizations) {
    // Back through the views to x, the blocked axis with them.
    int64_t axis = cast<IntegerAttr>(quant->getAttr("axis")).getInt();
    SmallVector<Operation *> views;
    Value value = quant->getOperand(0);
    while (Operation *view = value.getDefiningOp()) {
      if (auto transpose = dyn_cast<linalg::TransposeOp>(view)) {
        axis = transpose.getPermutation()[axis];
      } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(view)) {
        // The axis's group keeps it, with unit axes only.
        ReassociationIndices group = collapse.getReassociationIndices()[axis];
        auto source = collapse.getSrcType();
        int64_t kept = -1;
        for (int64_t member : group)
          if (source.getDimSize(member) != 1)
            kept = kept == -1 ? member : -2;
        if (kept == -2)
          break;
        axis = kept < 0 ? group.back() : kept;
      } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(view)) {
        // The axis is the last of its group and whole blocks long.
        int64_t group = -1;
        for (auto [index, members] :
             llvm::enumerate(expand.getReassociationIndices()))
          if (llvm::is_contained(members, axis))
            group = index;
        ReassociationIndices members = expand.getReassociationIndices()[group];
        if (members.back() != axis ||
            expand.getResultType().getDimSize(axis) % 32)
          break;
        axis = group;
      } else {
        break;
      }
      views.push_back(view);
      value = view->getOperand(0);
    }
    // A product's result stays written by its subgraph: a quantization of it
    // joined to it on the thread would read it there.
    auto type = cast<RankedTensorType>(value.getType());
    if (views.empty() || axis != type.getRank() - 1 || !type.hasStaticShape() ||
        isa_and_nonnull<linalg::MatmulOp, linalg::BatchMatmulOp>(
            value.getDefiningOp()) ||
        type.getShape().back() % 32 ||
        !llvm::all_of(quant->getUsers(), [&](Operation *user) {
          return user->getName().getStringRef() == "frontend.mxint8_to_fp32" &&
                 user->getAttr("axis") == quant->getAttr("axis");
        }))
      continue;
    // x quantized and decoded along its last axis, then the same views.
    OpBuilder b(quant);
    Location loc = quant->getLoc();
    SmallVector<int64_t> shape(type.getShape());
    shape.back() += shape.back() / 32;
    Attribute last = b.getI64IntegerAttr(axis);
    OperationState packState(loc, "frontend.fp32_to_mxint8");
    packState.addOperands(value);
    packState.addTypes(RankedTensorType::get(shape, b.getI8Type()));
    packState.addAttribute("axis", last);
    Operation *packed = b.create(packState);
    OperationState decodeState(loc, "frontend.mxint8_to_fp32");
    decodeState.addOperands(packed->getResult(0));
    decodeState.addTypes(type);
    decodeState.addAttribute("axis", last);
    Value decoded = b.create(decodeState)->getResult(0);
    IRMapping mapping;
    mapping.map(value, decoded);
    for (Operation *view : llvm::reverse(views))
      decoded = b.clone(*view, mapping)->getResult(0);
    for (Operation *user : llvm::make_early_inc_range(quant->getUsers())) {
      user->getResult(0).replaceAllUsesWith(decoded);
      user->erase();
    }
    quant->erase();
    for (Operation *view : views)
      if (view->use_empty())
        view->erase();
  }
}
