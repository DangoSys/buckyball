//===- ConvertLinalgToAcc.cpp - Toy hart recognition and lowering ---------===//
//
// convert-linalg-to-acc recognizes the quantized contractions of the toy
// hart (2-D convolutions and matrix products) and lowers them to VecBall.
//
// contraction_i8(x, input_step, w, weight_step, init) = init + sum_k x[m, k] *
// w[k, n] * input_step * weight_step[n], from a convolution or matmul of
// int8_to_fp32(x, input_step), the step per tensor, and int8_to_fp32(w,
// weight_step) (possibly transposed), the step per output channel. The
// contraction keeps its iteration space: `indexing_maps` (input, weight,
// output), `loop_ranges` and the loop dimensions of the matrix product,
// `m_dims` (batch and output pixels or rows), `n_dims` (output channels or
// columns) and `k_dims` (the reduction).
//
// It becomes C[M x N] = A[M x K] @ B[K x N]. The CPU packs A^T (im2col for a
// convolution) and B into zero-padded 16x16 INT8 tiles. For each 16x16 C
// tile, the C bank starts at zero and VecBall's matmul_at_i8 accumulates
// A^T and B one bank depth of K rows at a time. The CPU adds C *
// input_step * weight_step to the initial FP32 output.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "vectorOps.h"

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

// int8_to_fp32(codes, step) behind `value`, possibly through a transpose
// (moved onto the codes and the step): {codes, step}.
std::optional<std::pair<Value, Value>> quantized(OpBuilder &b, Value value) {
  auto transpose = value.getDefiningOp<linalg::TransposeOp>();
  Operation *dequantize = frontend(transpose ? transpose.getInput() : value,
                                   "frontend.int8_to_fp32");
  if (!dequantize)
    return std::nullopt;
  Value codes = dequantize->getOperand(0), step = dequantize->getOperand(1);
  if (!transpose)
    return std::make_pair(codes, step);
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  auto moved = [&](Value source) -> Value {
    auto type = cast<RankedTensorType>(source.getType());
    SmallVector<int64_t> shape;
    for (int64_t dim : permutation)
      shape.push_back(type.getDimSize(dim));
    Value init = tensor::EmptyOp::create(b, transpose.getLoc(), shape,
                                         type.getElementType());
    return linalg::TransposeOp::create(b, transpose.getLoc(), source, init,
                                       permutation)
        ->getResult(0);
  };
  return std::make_pair(moved(codes), moved(step));
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

void lowerContraction(::buddy::acc::CustomOp op) {
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
  int64_t groupBlocks = target::getTarget().bankDepth / 16;

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
  Value inputStep = scalar(inputs[1]);
  Value weightStep = buffer(b, loc, inputs[3]);
  ArrayRef<int64_t> stepShape =
      cast<RankedTensorType>(inputs[3].getType()).getShape();
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
  // The tile every C bank starts from: matmul_at_i8 reads the old C.
  Value zeroTile = memref::AllocOp::create(
      b, loc, MemRefType::get({1, 16, 16}, i32), b.getI64IntegerAttr(64));
  Value zero32 = arith::ConstantIntOp::create(b, loc, 0, 32);
  linalg::FillOp::create(b, loc, zero32, zeroTile);
  Value zero8 = arith::ConstantIntOp::create(b, loc, 0, 8);
  linalg::FillOp::create(b, loc, zero8, aTiles);
  linalg::FillOp::create(b, loc, zero8, bTiles);
  loops(b, loc, m, k, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, mDims}, {col, kDims}});
    Value v = memref::LoadOp::create(b, loc, input, element(maps[0], iv));
    // A^T: row k of a tile is column k of its 16 rows of A.
    memref::StoreOp::create(b, loc, v, aTiles,
                            tileIndex(row, col, kt, col, row));
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
  for (int64_t i = 0; i < mt; ++i)
    for (int64_t j = 0; j < nt; ++j) {
      Value cBank = BankMvinOp::create(
          b, loc, i64, zeroTile, BankAllocOp::create(b, loc, i64, 1, 4), 16, 1);
      // One bank depth of K rows of A^T and B per instruction.
      for (int64_t block = 0; block < kt; block += groupBlocks) {
        int64_t count = std::min(groupBlocks, kt - block);
        Value aBank = BankMvinOp::create(
            b, loc, i64, tile(aTiles, i * kt + block, count),
            BankAllocOp::create(b, loc, i64, 1, 1), count * 16, 1);
        Value bBank = BankMvinOp::create(
            b, loc, i64, tile(bTiles, j * kt + block, count),
            BankAllocOp::create(b, loc, i64, 1, 1), count * 16, 1);
        cBank = ::buddy::vector_ball::BankMatmulAtI8Op::create(
            b, loc, i64, aBank, bBank, cBank, b.getI64IntegerAttr(count * 16));
        BankReleaseOp::create(b, loc, aBank);
        BankReleaseOp::create(b, loc, bBank);
      }
      cBank = BankMvoutOp::create(b, loc, i64, tile(cTiles, i * nt + j, 1),
                                  cBank, 16, 1);
      BankReleaseOp::create(b, loc, cBank);
    }

  loops(b, loc, m, n, [&](Value row, Value col) {
    SmallVector<Value> iv = iterators({{row, mDims}, {col, nDims}});
    SmallVector<Value> at = element(maps[2], iv);
    Value sum = memref::LoadOp::create(b, loc, cTiles,
                                       tileIndex(row, col, nt, row, col));
    // The weight step of this output: the step tensor at the weight's
    // indices, its unit dimensions at 0.
    SmallVector<Value> stepAt;
    for (auto [index, size] : llvm::zip(element(maps[1], iv), stepShape))
      stepAt.push_back(size == 1 ? idx(0) : index);
    Value step = arith::MulFOp::create(
        b, loc, inputStep, memref::LoadOp::create(b, loc, weightStep, stepAt));
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
    return "Recognize the quantized contractions of the toy hart and lower "
           "them to VecBall";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AccDialect, ::buddy::vector_ball::VecBallDialect,
                    affine::AffineDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    SmallVector<linalg::LinalgOp> contractions;
    getOperation().walk([&](linalg::LinalgOp op) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
              linalg::Conv2DNchwFchwOp, linalg::MatmulOp>(op) &&
          op.hasPureTensorSemantics())
        contractions.push_back(op);
    });
    for (linalg::LinalgOp op : contractions) {
      OpBuilder builder(op);
      SmallVector<int64_t> m, n, k;
      if (isa<linalg::MatmulOp>(op)) {
        FailureOr<linalg::ContractionDimensions> dims =
            linalg::inferContractionDims(op);
        if (failed(dims) || !dims->batch.empty())
          continue;
        m.assign(dims->m.begin(), dims->m.end());
        n.assign(dims->n.begin(), dims->n.end());
        k.assign(dims->k.begin(), dims->k.end());
      } else {
        FailureOr<linalg::ConvolutionDimensions> dims =
            linalg::inferConvolutionDims(op);
        if (failed(dims) || !dims->depth.empty())
          continue;
        m.assign(dims->batch.begin(), dims->batch.end());
        m.append(dims->outputImage.begin(), dims->outputImage.end());
        n.assign(dims->outputChannel.begin(), dims->outputChannel.end());
        k.assign(dims->filterLoop.begin(), dims->filterLoop.end());
        k.append(dims->inputChannel.begin(), dims->inputChannel.end());
      }
      Operation *input =
          frontend(op.getDpsInputs()[0], "frontend.int8_to_fp32");
      if (!input || !perTensor(input->getOperand(1)))
        continue;
      auto weight = quantized(builder, op.getDpsInputs()[1]);
      if (!weight)
        continue;
      auto custom = CustomOp::create(
          builder, op.getLoc(), op->getResultTypes(), "contraction_i8",
          ValueRange{input->getOperand(0), input->getOperand(1), weight->first,
                     weight->second, op.getDpsInits()[0]});
      custom->setAttr("indexing_maps", op.getIndexingMaps());
      custom->setAttr("loop_ranges",
                      builder.getDenseI64ArrayAttr(op.getStaticLoopRanges()));
      custom->setAttr("m_dims", builder.getDenseI64ArrayAttr(m));
      custom->setAttr("n_dims", builder.getDenseI64ArrayAttr(n));
      custom->setAttr("k_dims", builder.getDenseI64ArrayAttr(k));
      Operation *filter = op.getDpsInputs()[1].getDefiningOp();
      op->replaceAllUsesWith(custom.getOutputs());
      op->erase();
      if (input->use_empty())
        input->erase();
      // The dequantization of the weight, behind its transpose if any.
      if (auto transpose = dyn_cast<linalg::TransposeOp>(filter)) {
        filter = transpose.getInput().getDefiningOp();
        if (transpose->use_empty())
          transpose->erase();
      }
      if (filter->use_empty())
        filter->erase();
    }

    SmallVector<::buddy::acc::CustomOp> computations;
    getOperation().walk(
        [&](::buddy::acc::CustomOp op) { computations.push_back(op); });
    for (auto op : computations) {
      if (op.getKind() != "contraction_i8") {
        op.emitOpError("has no bank lowering on this hart for ")
            << op.getKind();
        return signalPassFailure();
      }
      lowerContraction(op);
    }
  }
};

PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
