//===- ConvertLinalgToBuckyball.cpp - Toy core lowering -------------------===//
//
// A 2-D convolution becomes a matrix product C[M x N] = A[M x K] @ B[K x N]
// with M the batch and output pixels, N the output channels and K the filter
// window and input channels. The CPU packs A (im2col) and B into 16x16 INT8
// tiles, rounding the FP32 values to nearest even and saturating; Gemmini
// computes each C tile with matmul_i8_tile, accumulating over K; the CPU adds
// the INT32 tiles to the FP32 output.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
// Runs body(i, j) for i in [0, rows) and j in [0, cols).
void loops(OpBuilder &b, Location loc, int64_t rows, int64_t cols,
           function_ref<void(Value, Value)> body) {
  OpBuilder::InsertionGuard guard(b);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  auto outer = scf::ForOp::create(
      b, loc, zero, arith::ConstantIndexOp::create(b, loc, rows), one);
  b.setInsertionPointToStart(outer.getBody());
  auto inner = scf::ForOp::create(
      b, loc, zero, arith::ConstantIndexOp::create(b, loc, cols), one);
  b.setInsertionPointToStart(inner.getBody());
  body(outer.getInductionVar(), inner.getInductionVar());
}

LogicalResult lowerConvolution(linalg::LinalgOp op) {
  FailureOr<linalg::ConvolutionDimensions> conv =
      linalg::inferConvolutionDims(op);
  Value input = op.getDpsInputs()[0], filter = op.getDpsInputs()[1];
  Value output = op.getDpsInits()[0];
  if (failed(conv) || !conv->depth.empty() || !op.hasPureBufferSemantics() ||
      !getElementTypeOrSelf(input.getType()).isF32() ||
      !getElementTypeOrSelf(filter.getType()).isF32() ||
      !getElementTypeOrSelf(output.getType()).isF32())
    return op->emitOpError("toy lowers FP32 2-D convolutions on memrefs");
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  SmallVector<unsigned> mDims(conv->batch.begin(), conv->batch.end());
  SmallVector<unsigned> kDims(conv->filterLoop.begin(), conv->filterLoop.end());
  llvm::append_range(mDims, conv->outputImage);
  llvm::append_range(kDims, conv->inputChannel);
  ArrayRef<unsigned> nDims = conv->outputChannel;
  auto extent = [&](ArrayRef<unsigned> dims) {
    int64_t size = 1;
    for (unsigned dim : dims)
      size *= ranges[dim];
    return size;
  };
  int64_t m = extent(mDims), n = extent(nDims), k = extent(kDims);
  int64_t mt = (m + 15) / 16, nt = (n + 15) / 16, kt = (k + 15) / 16;

  OpBuilder b(op);
  Location loc = op.getLoc();
  Type i8 = b.getI8Type(), i32 = b.getI32Type(), i64 = b.getI64Type();
  auto idx = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  auto f32 = [&](float v) -> Value {
    return arith::ConstantOp::create(b, loc, b.getF32FloatAttr(v));
  };
  // The loop dimension values of flat indices along each dimension group.
  auto iterators = [&](ArrayRef<std::pair<Value, ArrayRef<unsigned>>> flats) {
    SmallVector<Value> iv(ranges.size(), idx(0));
    for (auto [flat, dims] : flats)
      for (unsigned dim : llvm::reverse(dims)) {
        iv[dim] = arith::RemUIOp::create(b, loc, flat, idx(ranges[dim]));
        flat = arith::DivUIOp::create(b, loc, flat, idx(ranges[dim]));
      }
    return iv;
  };
  auto element = [&](OpOperand &operand, ArrayRef<Value> iv) {
    AffineMap map = op.getMatchingIndexingMap(&operand);
    SmallVector<Value> indices;
    for (AffineExpr result : map.getResults())
      indices.push_back(affine::AffineApplyOp::create(
          b, loc, AffineMap::get(map.getNumDims(), 0, result), iv));
    return indices;
  };
  auto quantize = [&](Value v) -> Value {
    v = arith::MaximumFOp::create(b, loc, v, f32(-128));
    v = arith::MinimumFOp::create(b, loc, v, f32(127));
    v = math::RoundEvenOp::create(b, loc, v);
    return arith::FPToSIOp::create(b, loc, i8, v);
  };
  // Tile (row / 16) * stride + col / 16, element (row % 16, col % 16).
  auto tileIndex = [&](Value row, Value col, int64_t stride) {
    Value sixteen = idx(16);
    Value tile = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(
            b, loc, arith::DivUIOp::create(b, loc, row, sixteen), idx(stride)),
        arith::DivUIOp::create(b, loc, col, sixteen));
    return SmallVector<Value>{tile,
                              arith::RemUIOp::create(b, loc, row, sixteen),
                              arith::RemUIOp::create(b, loc, col, sixteen)};
  };

  Value aTiles = memref::AllocOp::create(
      b, loc, MemRefType::get({mt * kt, 16, 16}, i8), b.getI64IntegerAttr(64));
  Value bTiles = memref::AllocOp::create(
      b, loc, MemRefType::get({kt * nt, 16, 16}, i8), b.getI64IntegerAttr(64));
  Value cTiles = memref::AllocOp::create(
      b, loc, MemRefType::get({mt * nt, 16, 16}, i32), b.getI64IntegerAttr(64));
  Value zero8 = arith::ConstantIntOp::create(b, loc, 0, 8);
  linalg::FillOp::create(b, loc, zero8, aTiles);
  linalg::FillOp::create(b, loc, zero8, bTiles);
  loops(b, loc, m, k, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, mDims}, {col, kDims}});
    Value v =
        memref::LoadOp::create(b, loc, input, element(op->getOpOperand(0), iv));
    memref::StoreOp::create(b, loc, quantize(v), aTiles,
                            tileIndex(row, col, kt));
  });
  loops(b, loc, k, n, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, kDims}, {col, nDims}});
    Value v = memref::LoadOp::create(b, loc, filter,
                                     element(op->getOpOperand(1), iv));
    memref::StoreOp::create(b, loc, quantize(v), bTiles,
                            tileIndex(row, col, nt));
  });

  auto tile = [&](Value tiles, int64_t index) -> Value {
    return memref::SubViewOp::create(
        b, loc, tiles, ArrayRef<int64_t>{index, 0, 0},
        ArrayRef<int64_t>{1, 16, 16}, ArrayRef<int64_t>{1, 1, 1});
  };
  Value base = arith::ConstantIntOp::create(b, loc, 0, 64);
  Value aBank = BankAllocOp::create(b, loc, i64, 1, 1);
  Value bBank = BankAllocOp::create(b, loc, i64, 1, 1);
  for (int64_t i = 0; i < mt; ++i)
    for (int64_t j = 0; j < nt; ++j) {
      Value cBank = BankAllocOp::create(b, loc, i64, 1, 4);
      for (int64_t r = 0; r < kt; ++r) {
        aBank = BankMvinOp::create(b, loc, i64, tile(aTiles, i * kt + r), aBank,
                                   16, 1);
        bBank = BankMvinOp::create(b, loc, i64, tile(bTiles, r * nt + j), bBank,
                                   16, 1);
        auto matmul = BankCustomOp::create(
            b, loc, i64, ValueRange{aBank, bBank}, cBank,
            ValueRange{base, base, base}, "matmul_i8_tile", r > 0);
        if (r > 0)
          matmul->setAttr("shift", b.getI64IntegerAttr(0));
        cBank = matmul.getDestinationOut();
      }
      cBank = BankMvoutOp::create(b, loc, i64, tile(cTiles, i * nt + j), cBank,
                                  16, 1);
      BankReleaseOp::create(b, loc, cBank);
    }
  BankReleaseOp::create(b, loc, aBank);
  BankReleaseOp::create(b, loc, bBank);

  loops(b, loc, m, n, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, mDims}, {col, nDims}});
    SmallVector<Value> at = element(op->getOpOperand(2), iv);
    Value sum = memref::LoadOp::create(b, loc, cTiles, tileIndex(row, col, nt));
    Value result = arith::AddFOp::create(
        b, loc, memref::LoadOp::create(b, loc, output, at),
        arith::SIToFPOp::create(b, loc, b.getF32Type(), sum));
    memref::StoreOp::create(b, loc, result, output, at);
  });
  for (Value buffer : {aTiles, bTiles, cTiles})
    memref::DeallocOp::create(b, loc, buffer);
  op->erase();
  return success();
}

class ConvertLinalgToBuckyballPass
    : public PassWrapper<ConvertLinalgToBuckyballPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToBuckyballPass)
  StringRef getArgument() const final { return "convert-linalg-to-buckyball"; }
  StringRef getDescription() const final {
    return "Lower 2-D convolutions to the toy core bank dataflow";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BuckyballDialect, affine::AffineDialect, arith::ArithDialect,
                math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    SmallVector<linalg::LinalgOp> convolutions;
    getOperation().walk([&](linalg::LinalgOp op) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
              linalg::Conv2DNchwFchwOp>(op))
        convolutions.push_back(op);
    });
    for (linalg::LinalgOp op : convolutions)
      if (failed(lowerConvolution(op)))
        return signalPassFailure();
  }
};

PassRegistration<ConvertLinalgToBuckyballPass> registration;
} // namespace
