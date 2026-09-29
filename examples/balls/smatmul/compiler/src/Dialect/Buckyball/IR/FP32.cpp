#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace mlir::bufferization;
using namespace buddy::buckyball;

bool FP32MatmulOp::bufferizesToAllocation(Value) { return true; }
bool FP32MatmulOp::bufferizesToMemoryRead(OpOperand &, const AnalysisState &) {
  return true;
}
bool FP32MatmulOp::bufferizesToMemoryWrite(OpOperand &, const AnalysisState &) {
  return false;
}
AliasingValueList FP32MatmulOp::getAliasingValues(OpOperand &,
                                                  const AnalysisState &) {
  return {};
}
LogicalResult FP32MatmulOp::bufferize(RewriterBase &b,
                                      const BufferizationOptions &options,
                                      BufferizationState &state) {
  auto a = getBuffer(b, getLhs(), options, state);
  auto w = getBuffer(b, getRhs(), options, state);
  if (failed(a) || failed(w))
    return failure();
  auto type = cast<RankedTensorType>(getOutput().getType());
  if (!type.hasStaticShape())
    return emitError("requires static FP32 output shape");
  Value out = b.create<memref::AllocOp>(
      getLoc(), MemRefType::get(type.getShape(), type.getElementType()));
  b.create<FP32MemMatmulOp>(getLoc(), *a, *w, out);
  replaceOpWithBufferizedValues(b, getOperation(), out);
  return success();
}
