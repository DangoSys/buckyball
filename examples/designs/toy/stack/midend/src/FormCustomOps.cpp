//===- FormCustomOps.cpp - Recognize the toy core computations ------------===//
//
// conv2d_i8(x, input_step, w, weight_step, init) = init + conv(x, w) *
// input_step * weight_step, from a 2-D convolution of
// int8_to_fp32(fp32_to_int8(x', multiplier), input_step) and
// int8_to_fp32(w, weight_step), both steps per tensor. The convolution keeps
// its iteration space: `indexing_maps` (input, filter, output), `loop_ranges`
// and the loop dimensions of the matrix product, `m_dims` (batch and output
// pixels), `n_dims` (output channels) and `k_dims` (filter window and input
// channels).
//
// The core forms it only when its banks fit: the C tile takes four groups,
// bank 0 is the zero bank, and A and B each take a group per bank depth of K.
//
//===----------------------------------------------------------------------===//

#include "Custom/CustomOps.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using ::buddy::custom::CustomOp;

namespace {
Operation *frontend(Value value, StringRef name) {
  Operation *op = value.getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}

bool perTensor(Value scale) {
  return llvm::all_of(cast<RankedTensorType>(scale.getType()).getShape(),
                      [](int64_t size) { return size == 1; });
}

class FormCustomOpsPass
    : public PassWrapper<FormCustomOpsPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormCustomOpsPass)
  StringRef getArgument() const final { return "form-custom-ops"; }
  StringRef getDescription() const final {
    return "Recognize the quantized convolutions of the toy core";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::custom::CustomDialect>();
  }
  void runOnOperation() override {
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t maxK = (target.bankNum - 5) / 2 * target.bankDepth;
    SmallVector<linalg::LinalgOp> convolutions;
    getOperation().walk([&](linalg::LinalgOp op) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
              linalg::Conv2DNchwFchwOp>(op) &&
          op.hasPureTensorSemantics())
        convolutions.push_back(op);
    });
    for (linalg::LinalgOp op : convolutions) {
      Operation *input =
          frontend(op.getDpsInputs()[0], "frontend.int8_to_fp32");
      Operation *weight =
          frontend(op.getDpsInputs()[1], "frontend.int8_to_fp32");
      if (!input || !weight)
        continue;
      Operation *quantize =
          frontend(input->getOperand(0), "frontend.fp32_to_int8");
      FailureOr<linalg::ConvolutionDimensions> conv =
          linalg::inferConvolutionDims(op);
      if (!quantize || !perTensor(input->getOperand(1)) ||
          !perTensor(weight->getOperand(1)) || failed(conv) ||
          !conv->depth.empty())
        continue;
      SmallVector<int64_t> ranges = op.getStaticLoopRanges();
      int64_t k = 1;
      for (unsigned dim : conv->filterLoop)
        k *= ranges[dim];
      for (unsigned dim : conv->inputChannel)
        k *= ranges[dim];
      if (k > maxK)
        continue;
      OpBuilder builder(op);
      auto dims = [&](ArrayRef<unsigned> first, ArrayRef<unsigned> second) {
        SmallVector<int64_t> all(first.begin(), first.end());
        all.append(second.begin(), second.end());
        return builder.getDenseI64ArrayAttr(all);
      };
      auto custom = CustomOp::create(
          builder, op.getLoc(), op->getResultTypes(), "conv2d_i8",
          ValueRange{quantize->getResult(0), input->getOperand(1),
                     weight->getOperand(0), weight->getOperand(1),
                     op.getDpsInits()[0]});
      custom->setAttr("indexing_maps", op.getIndexingMaps());
      custom->setAttr("loop_ranges", builder.getDenseI64ArrayAttr(ranges));
      custom->setAttr("m_dims", dims(conv->batch, conv->outputImage));
      custom->setAttr("n_dims", dims(conv->outputChannel, {}));
      custom->setAttr("k_dims", dims(conv->filterLoop, conv->inputChannel));
      op->replaceAllUsesWith(custom.getOutputs());
      op->erase();
      for (Operation *dequantize : {input, weight})
        if (dequantize->use_empty())
          dequantize->erase();
    }
  }
};

PassRegistration<FormCustomOpsPass> registration;
} // namespace
