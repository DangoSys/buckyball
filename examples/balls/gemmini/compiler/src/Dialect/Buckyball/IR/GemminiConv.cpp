#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace mlir::bufferization;
using namespace buddy::buckyball;
bool GemminiConvOp::bufferizesToAllocation(Value) { return true; }
bool GemminiConvOp::bufferizesToMemoryRead(OpOperand &, const AnalysisState &) {
  return true;
}
bool GemminiConvOp::bufferizesToMemoryWrite(OpOperand &,
                                            const AnalysisState &) {
  return false;
}
AliasingValueList GemminiConvOp::getAliasingValues(OpOperand &,
                                                   const AnalysisState &) {
  return {};
}
LogicalResult GemminiConvOp::bufferize(RewriterBase &b,
                                       const BufferizationOptions &options,
                                       BufferizationState &state) {
  auto input = getBuffer(b, getInput(), options, state),
       weight = getBuffer(b, getWeight(), options, state),
       bias = getBuffer(b, getBias(), options, state);
  if (failed(input) || failed(weight) || failed(bias))
    return failure();
  auto type = cast<RankedTensorType>(getOutput().getType());
  if (!type.hasStaticShape())
    return emitError("Gemmini requires static model shapes");
  Value output = b.create<memref::AllocOp>(
      getLoc(), MemRefType::get(type.getShape(), type.getElementType()));
  b.create<GemminiMemConvOp>(getLoc(), *input, *weight, *bias, output,
                             getKhAttr(), getKwAttr(), getStrideAttr(),
                             getPaddingAttr(), getInputScaleAttr(),
                             getWeightScaleAttr());
  replaceOpWithBufferizedValues(b, getOperation(), output);
  return success();
}
