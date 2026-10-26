#include "Conversion/LowerLinalgToBuckyball/LowerLinalgToBuckyball.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace {
class FormGobanMegaKernelsPass
    : public PassWrapper<FormGobanMegaKernelsPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormGobanMegaKernelsPass)
  StringRef getArgument() const final { return "form-goban-mega-kernels"; }
  StringRef getDescription() const final {
    return "Fuse quantized matrix operators within one Goban core";
  }
  void runOnOperation() override {
    DenseMap<Block *, SmallVector<linalg::GenericOp>> blocks;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op->hasAttr("buckyball.quantized_matmul"))
        blocks[op->getBlock()].push_back(op);
    });
    for (auto &entry : blocks) {
      mlir::buddy::prepareQuantizedWeights(entry.second);
      mlir::buddy::formMegaKernelRegions(entry.second);
    }
  }
};
[[maybe_unused]] const bool registered = [] {
  PassRegistration<FormGobanMegaKernelsPass>();
  return true;
}();
} // namespace
