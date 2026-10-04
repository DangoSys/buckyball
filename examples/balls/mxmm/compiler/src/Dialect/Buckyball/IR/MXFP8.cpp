#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
using namespace mlir::bufferization;
using namespace ::buddy::buckyball;

namespace {
FailureOr<int64_t> packedSize(int64_t m, int64_t k, int64_t rows, int64_t chunk,
                              int64_t stride) {
  if (m <= 0 || k <= 0 || k % 32 || rows <= 0 || (rows != 1 && rows % 16) ||
      chunk <= 0 || chunk % 32 || chunk >= 4096 || stride <= 0 ||
      stride > 65536 || !llvm::isPowerOf2_64(stride) ||
      rows > stride * 32 / (chunk * 33))
    return failure();
  int64_t panels, bytes;
  if (llvm::MulOverflow((m - 1) / rows + 1, (k - 1) / chunk + 1, panels) ||
      llvm::MulOverflow(panels, stride, bytes))
    return failure();
  return bytes;
}
} // namespace

LogicalResult MXFP8QuantOp::verify() {
  auto input = cast<RankedTensorType>(getInput().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  if (!input.hasStaticShape() || input.getRank() != 2 ||
      !input.getElementType().isF32() || !output.hasStaticShape() ||
      output.getRank() != 1 || !output.getElementType().isInteger(8))
    return emitOpError("requires a static FP32 matrix and packed byte tensor");
  auto bytes = packedSize(input.getShape()[0], input.getShape()[1],
                          getTileRows(), getTileK(), getBankBytes());
  if (failed(bytes) || output.getShape()[0] != *bytes)
    return emitOpError("activation panel layout or packed byte shape mismatch");
  return success();
}

bool MXFP8QuantOp::bufferizesToAllocation(Value) { return true; }
bool MXFP8QuantOp::bufferizesToMemoryRead(OpOperand &, const AnalysisState &) {
  return true;
}
bool MXFP8QuantOp::bufferizesToMemoryWrite(OpOperand &, const AnalysisState &) {
  return false;
}
AliasingValueList MXFP8QuantOp::getAliasingValues(OpOperand &,
                                                  const AnalysisState &) {
  return {};
}
LogicalResult MXFP8QuantOp::bufferize(RewriterBase &b,
                                      const BufferizationOptions &options,
                                      BufferizationState &state) {
  auto input = getBuffer(b, getInput(), options, state);
  if (failed(input))
    return failure();
  auto type = cast<RankedTensorType>(getOutput().getType());
  auto output = b.create<memref::AllocOp>(
      getLoc(), MemRefType::get(type.getShape(), type.getElementType()));
  output->setAttr("alignment", b.getI64IntegerAttr(getBankBytes()));
  auto dynamic = ShapedType::kDynamic;
  auto inputType = MemRefType::get(
      {dynamic, dynamic}, b.getF32Type(),
      StridedLayoutAttr::get(getContext(), dynamic, {dynamic, dynamic}));
  auto bytesType = MemRefType::get({dynamic}, b.getI8Type());
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<func::FuncOp>("mxfp8_quant")) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    auto function = b.create<func::FuncOp>(
        getLoc(), "mxfp8_quant",
        b.getFunctionType({inputType, bytesType, b.getI64Type(), b.getI64Type(),
                           b.getI64Type()},
                          {}));
    function.setPrivate();
    function->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  }
  SmallVector<Value> arguments{
      b.create<memref::CastOp>(getLoc(), inputType, *input),
      b.create<memref::CastOp>(getLoc(), bytesType, output)};
  for (int64_t value : {getTileRows(), getTileK(), getBankBytes()})
    arguments.push_back(b.create<arith::ConstantIntOp>(getLoc(), value, 64));
  b.create<func::CallOp>(getLoc(), "mxfp8_quant", TypeRange{}, arguments);
  replaceOpWithBufferizedValues(b, getOperation(),
                                ValueRange{output.getResult()});
  return success();
}

LogicalResult MXFP8MatmulOp::verify() {
  auto input = cast<RankedTensorType>(getInput().getType());
  auto weight = cast<RankedTensorType>(getWeight().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  if (!input.hasStaticShape() || input.getRank() != 1 ||
      !input.getElementType().isInteger(8) || !weight.hasStaticShape() ||
      weight.getRank() != 1 || !weight.getElementType().isInteger(8) ||
      !output.hasStaticShape() || output.getRank() != 2 ||
      !output.getElementType().isF32())
    return emitOpError(
        "requires packed activation/weight bytes and a static FP32 matrix");
  int64_t m = output.getShape()[0], n = output.getShape()[1],
          k = getReductionK();
  int64_t rows = m == 1 ? 1 : getTileM(), columns = getTileN();
  auto activationBytes = packedSize(m, k, rows, getTileK(), getBankBytes());
  auto weightBytes = packedSize(n, k, columns, getTileK(), getBankBytes());
  if (columns <= 0 || columns % 16 || failed(activationBytes) ||
      failed(weightBytes) || input.getShape()[0] != *activationBytes ||
      weight.getShape()[0] != *weightBytes)
    return emitOpError("logical K, panel layout or packed byte shape mismatch");
  if (auto quant = getInput().getDefiningOp<MXFP8QuantOp>()) {
    auto source = cast<RankedTensorType>(quant.getInput().getType());
    if (source.getRank() != 2 || source.getShape()[0] != m ||
        source.getShape()[1] != k)
      return emitOpError("activation quantization matrix shape mismatch");
    if (quant.getTileRows() != rows || quant.getTileK() != getTileK() ||
        quant.getBankBytes() != getBankBytes())
      return emitOpError("activation quantization panel layout mismatch");
  }
  return success();
}

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
  auto input = getBuffer(b, getInput(), options, state);
  auto weight = getBuffer(b, getWeight(), options, state);
  if (failed(input) || failed(weight))
    return failure();
  auto type = cast<RankedTensorType>(getOutput().getType());
  auto output = b.create<memref::AllocOp>(
      getLoc(), MemRefType::get(type.getShape(), type.getElementType()));
  b.create<MXFP8MemMatmulOp>(
      getLoc(), *input, *weight, output, getReductionKAttr(), getTileKAttr(),
      getTileMAttr(), getTileNAttr(), getBankBytesAttr());
  replaceOpWithBufferizedValues(b, getOperation(),
                                ValueRange{output.getResult()});
  return success();
}
