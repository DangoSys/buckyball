#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace mlir::bufferization;
using namespace buddy::buckyball;

bool MXFP8MatmulOp::bufferizesToAllocation(Value) { return true; }
bool MXFP8MatmulOp::bufferizesToMemoryRead(OpOperand &, const AnalysisState &) {
  return true;
}
bool MXFP8MatmulOp::bufferizesToMemoryWrite(OpOperand &,
                                            const AnalysisState &) {
  return false;
}
AliasingValueList MXFP8MatmulOp::getAliasingValues(OpOperand &,
                                                   const AnalysisState &) {
  return {};
}
LogicalResult MXFP8MatmulOp::bufferize(RewriterBase &b,
                                       const BufferizationOptions &options,
                                       BufferizationState &state) {
  auto a = getBuffer(b, getInput(), options, state);
  auto w = getBuffer(b, getWeight(), options, state);
  if (failed(a) || failed(w))
    return failure();
  auto type = cast<RankedTensorType>(getOutput().getType());
  if (!type.hasStaticShape())
    return emitError("requires static MXFP8 output shape");
  Value out = b.create<memref::AllocOp>(
      getLoc(), MemRefType::get(type.getShape(), type.getElementType()));
  b.create<MXFP8MemMatmulOp>(getLoc(), *a, *w, out);
  replaceOpWithBufferizedValues(b, getOperation(), out);
  return success();
}
