//===- ConvertLinalgToAcc.cpp - Toy core recognition and lowering ---------===//
//
// convert-linalg-to-acc recognizes the quantized convolutions of the toy core
// and lowers them to its bank dataflow.
//
// conv2d_i8(x, input_step, w, weight_step, init) = init + conv(x, w) *
// input_step * weight_step, from a 2-D convolution of
// int8_to_fp32(fp32_to_int8(x', multiplier), input_step) and
// int8_to_fp32(w, weight_step), both steps per tensor. The convolution keeps
// its iteration space: `indexing_maps` (input, filter, output), `loop_ranges`
// and the loop dimensions of the matrix product, `m_dims` (batch and output
// pixels), `n_dims` (output channels) and `k_dims` (filter window and input
// channels). The core forms it only when its banks fit: the C tile takes four
// groups, bank 0 is the zero bank, and A and B each take a group per bank
// depth of K.
//
// conv2d_i8 becomes a matrix product C[M x N] = A[M x K] @ B[K x N] over
// those dimension groups. The CPU packs A (im2col) and B into zero-padded
// 16x16 INT8 tiles. For each 16x16 C tile, the A bank takes the K blocks of
// the tile's rows and the B bank the K rows of its columns, one bank-deep
// group at a time gathered with bank_transfer, and Gemmini's matmul_i8_tile
// computes C. The CPU adds C * input_step * weight_step to the initial FP32
// output.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "gemminiOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::buddy::acc;

namespace {
Operation *frontend(Value value, StringRef name) {
  Operation *op = value.getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}

bool perTensor(Value scale) {
  return llvm::all_of(cast<RankedTensorType>(scale.getType()).getShape(),
                      [](int64_t size) { return size == 1; });
}

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

// The buffer of a contiguous tensor.
Value buffer(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto memory = MemRefType::get(
      type.getShape(), type.getElementType(),
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                             computeSuffixProduct(type.getShape())));
  return bufferization::ToBufferOp::create(b, loc, memory, tensor, true);
}

void lowerConvolution(::buddy::acc::CustomOp op) {
  SmallVector<AffineMap> maps = llvm::map_to_vector(
      cast<ArrayAttr>(op->getAttr("indexing_maps")),
      [](Attribute map) { return cast<AffineMapAttr>(map).getValue(); });
  auto array = [&](StringRef name) {
    return cast<DenseI64ArrayAttr>(op->getAttr(name)).asArrayRef();
  };
  ArrayRef<int64_t> ranges = array("loop_ranges");
  ArrayRef<int64_t> mDims = array("m_dims"), nDims = array("n_dims"),
                    kDims = array("k_dims");
  auto extent = [&](ArrayRef<int64_t> dims) {
    int64_t size = 1;
    for (int64_t dim : dims)
      size *= ranges[dim];
    return size;
  };
  int64_t m = extent(mDims), n = extent(nDims), k = extent(kDims);
  int64_t mt = (m + 15) / 16, nt = (n + 15) / 16, kt = (k + 15) / 16;
  // The K blocks a bank group holds.
  int64_t groupBlocks = buckyball_target::getBuckyballTarget().bankDepth / 16;

  OpBuilder b(op);
  Location loc = op.getLoc();
  Type i8 = b.getI8Type(), i32 = b.getI32Type(), i64 = b.getI64Type();
  auto idx = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  // The loop dimension values of flat indices along each dimension group.
  auto iterators = [&](ArrayRef<std::pair<Value, ArrayRef<int64_t>>> flats) {
    SmallVector<Value> iv(ranges.size(), idx(0));
    for (auto [flat, dims] : flats)
      for (int64_t dim : llvm::reverse(dims)) {
        iv[dim] = arith::RemUIOp::create(b, loc, flat, idx(ranges[dim]));
        flat = arith::DivUIOp::create(b, loc, flat, idx(ranges[dim]));
      }
    return iv;
  };
  auto element = [&](AffineMap map, ArrayRef<Value> iv) {
    SmallVector<Value> indices;
    for (AffineExpr result : map.getResults())
      indices.push_back(affine::AffineApplyOp::create(
          b, loc, AffineMap::get(map.getNumDims(), 0, result), iv));
    return indices;
  };
  // Tile (outer / 16) * stride + inner / 16, element (row % 16, col % 16).
  auto tileIndex = [&](Value outer, Value inner, int64_t stride, Value row,
                       Value col) {
    Value sixteen = idx(16);
    Value tile = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(b, loc,
                              arith::DivUIOp::create(b, loc, outer, sixteen),
                              idx(stride)),
        arith::DivUIOp::create(b, loc, inner, sixteen));
    return SmallVector<Value>{tile,
                              arith::RemUIOp::create(b, loc, row, sixteen),
                              arith::RemUIOp::create(b, loc, col, sixteen)};
  };
  auto scalar = [&](Value scale) -> Value {
    SmallVector<Value> zeros(cast<RankedTensorType>(scale.getType()).getRank(),
                             idx(0));
    return tensor::ExtractOp::create(b, loc, scale, zeros);
  };

  ValueRange inputs = op.getInputs();
  Value input = buffer(b, loc, inputs[0]), filter = buffer(b, loc, inputs[2]);
  Value step =
      arith::MulFOp::create(b, loc, scalar(inputs[1]), scalar(inputs[3]));
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(resultType.getShape(), b.getF32Type()));
  memref::CopyOp::create(b, loc, buffer(b, loc, inputs[4]), output);

  Value aTiles = memref::AllocOp::create(
      b, loc, MemRefType::get({mt * kt, 16, 16}, i8), b.getI64IntegerAttr(64));
  // B tiles by column: tile j * kt + r holds K rows 16r .. 16r + 15.
  Value bTiles = memref::AllocOp::create(
      b, loc, MemRefType::get({nt * kt, 16, 16}, i8), b.getI64IntegerAttr(64));
  Value cTiles = memref::AllocOp::create(
      b, loc, MemRefType::get({mt * nt, 16, 16}, i32), b.getI64IntegerAttr(64));
  Value zero8 = arith::ConstantIntOp::create(b, loc, 0, 8);
  linalg::FillOp::create(b, loc, zero8, aTiles);
  linalg::FillOp::create(b, loc, zero8, bTiles);
  loops(b, loc, m, k, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, mDims}, {col, kDims}});
    Value v = memref::LoadOp::create(b, loc, input, element(maps[0], iv));
    memref::StoreOp::create(b, loc, v, aTiles,
                            tileIndex(row, col, kt, row, col));
  });
  loops(b, loc, k, n, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, kDims}, {col, nDims}});
    Value v = memref::LoadOp::create(b, loc, filter, element(maps[1], iv));
    memref::StoreOp::create(b, loc, v, bTiles,
                            tileIndex(col, row, kt, row, col));
  });

  // `count` consecutive tiles from `index`.
  auto tile = [&](Value tiles, int64_t index, int64_t count) -> Value {
    return memref::SubViewOp::create(
        b, loc, tiles, ArrayRef<int64_t>{index, 0, 0},
        ArrayRef<int64_t>{count, 16, 16}, ArrayRef<int64_t>{1, 1, 1});
  };
  // The kt tiles from `first` in one bank, a group of groupBlocks tiles at a
  // time.
  auto gather = [&](Value tiles, int64_t first) {
    Value bank;
    for (int64_t block = 0; block < kt; block += groupBlocks) {
      int64_t count = std::min(groupBlocks, kt - block);
      Value group = BankMvinOp::create(
          b, loc, i64, tile(tiles, first + block, count),
          BankAllocOp::create(b, loc, i64, 1, 1), count * 16, 1);
      bank = block
                 ? BankTransferOp::create(b, loc, i64, group, bank).getResult()
                 : group;
    }
    return bank;
  };
  for (int64_t i = 0; i < mt; ++i)
    for (int64_t j = 0; j < nt; ++j) {
      Value aBank = gather(aTiles, i * kt);
      Value bBank = gather(bTiles, j * kt);
      Value cBank = BankAllocOp::create(b, loc, i64, 1, 4);
      cBank = ::buddy::gemmini_ball::BankMatmulI8TileOp::create(
          b, loc, i64, aBank, bBank, cBank, b.getI64IntegerAttr(kt * 16),
          UnitAttr(), IntegerAttr());
      cBank = BankMvoutOp::create(b, loc, i64, tile(cTiles, i * nt + j, 1),
                                  cBank, 16, 1);
      BankReleaseOp::create(b, loc, aBank);
      BankReleaseOp::create(b, loc, bBank);
      BankReleaseOp::create(b, loc, cBank);
    }

  loops(b, loc, m, n, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, mDims}, {col, nDims}});
    SmallVector<Value> at = element(maps[2], iv);
    Value sum = memref::LoadOp::create(b, loc, cTiles,
                                       tileIndex(row, col, nt, row, col));
    Value scaled = arith::MulFOp::create(
        b, loc, arith::SIToFPOp::create(b, loc, b.getF32Type(), sum), step);
    Value result = arith::AddFOp::create(
        b, loc, memref::LoadOp::create(b, loc, output, at), scaled);
    memref::StoreOp::create(b, loc, result, output, at);
  });
  op.getResult(0).replaceAllUsesWith(bufferization::ToTensorOp::create(
      b, loc, resultType, output, true, true));
  op.erase();
}

class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Recognize the quantized convolutions of the toy core and lower "
           "them to its bank dataflow";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AccDialect, ::buddy::gemmini_ball::GemminiBallDialect,
                    affine::AffineDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
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

    SmallVector<::buddy::acc::CustomOp> computations;
    getOperation().walk(
        [&](::buddy::acc::CustomOp op) { computations.push_back(op); });
    for (auto op : computations) {
      if (op.getKind() != "conv2d_i8") {
        op.emitOpError("has no bank lowering on this core for ")
            << op.getKind();
        return signalPassFailure();
      }
      lowerConvolution(op);
    }
  }
};

PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
