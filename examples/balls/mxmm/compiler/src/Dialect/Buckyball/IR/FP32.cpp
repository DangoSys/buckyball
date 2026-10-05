#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace mlir::bufferization;
using namespace ::buddy::buckyball;

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
  Value rhs = *w;
  if (getRhsTransposedAttr()) {
    auto source = cast<MemRefType>(rhs.getType());
    int rank = source.getRank();
    if (rank != 2 && rank != 3)
      return emitError("transposed FP32 RHS requires rank 2 or 3");
    SmallVector<unsigned> permutation;
    for (int index = 0; index < rank; ++index)
      permutation.push_back(index);
    std::swap(permutation[rank - 2], permutation[rank - 1]);
    auto map = AffineMap::getPermutationMap(permutation, getContext());
    rhs = b.create<memref::TransposeOp>(getLoc(), rhs, AffineMapAttr::get(map));
  }
  b.create<FP32MemMatmulOp>(getLoc(), *a, rhs, out, getFusedAttr());
  replaceOpWithBufferizedValues(b, getOperation(), out);
  return success();
}
