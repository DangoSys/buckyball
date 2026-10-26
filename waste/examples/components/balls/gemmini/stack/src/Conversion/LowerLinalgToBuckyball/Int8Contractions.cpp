//===- Int8Contractions.cpp - INT8 contractions for GemminiBall
//------------===//

#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class Int8ContractionLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto convolution = op->getAttrOfType<DictionaryAttr>("buddy.int8_conv");
    bool matrix = op->hasAttr("buddy.int8_matmul");
    if (!convolution && !matrix)
      return failure();
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return op.emitError(
          "INT8 contraction requires two inputs and one output");
    Value input = op.getInputs()[0], weight = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    auto in = dyn_cast<MemRefType>(input.getType());
    auto w = dyn_cast<MemRefType>(weight.getType());
    auto out = dyn_cast<MemRefType>(output.getType());
    if (!in || !w || !out || !in.hasStaticShape() || !w.hasStaticShape() ||
        !out.hasStaticShape() || !in.getElementType().isInteger(8) ||
        !w.getElementType().isInteger(8) || !out.getElementType().isInteger(32))
      return op.emitError(
          "GemminiBall requires static INT8 operands and INT32 output");
    Location loc = op.getLoc();
    Value packedWeight;
    SmallVector<int64_t> permutation;
    if (matrix) {
      auto map = op.getIndexingMapsArray()[1];
      if (map.getResult(0) == getAffineDimExpr(1, rewriter.getContext()))
        permutation = {1, 0};
    } else {
      auto layout =
          cast<DenseI64ArrayAttr>(convolution.get("weight_permutation"))
              .asArrayRef();
      if (layout == ArrayRef<int64_t>{0, 1, 2, 3})
        permutation = {2, 3, 1, 0};
    }
    if (!permutation.empty()) {
      SmallVector<int64_t> shape;
      for (int64_t axis : permutation)
        shape.push_back(w.getShape()[axis]);
      packedWeight = memref::AllocOp::create(
          rewriter, loc, MemRefType::get(shape, rewriter.getI8Type()));
      SmallVector<Value> coordinates;
      for (int64_t extent : w.getShape()) {
        auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto end = arith::ConstantIndexOp::create(rewriter, loc, extent);
        auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto loop = scf::ForOp::create(rewriter, loc, zero, end, one);
        rewriter.setInsertionPointToStart(loop.getBody());
        coordinates.push_back(loop.getInductionVar());
      }
      Value value = memref::LoadOp::create(rewriter, loc, weight, coordinates);
      SmallVector<Value> indices;
      for (int64_t axis : permutation)
        indices.push_back(coordinates[axis]);
      memref::StoreOp::create(rewriter, loc, value, packedWeight, indices);
      rewriter.setInsertionPoint(op);
      weight = packedWeight;
      w = cast<MemRefType>(weight.getType());
    }
    int64_t kh = 1, kw = 1, stride = 1, channels;
    if (matrix) {
      if (in.getRank() != 2 || w.getRank() != 2 || out.getRank() != 2)
        return op.emitError("GemminiBall matrix operands must have rank two");
      channels = out.getShape()[1];
      input = memref::ExpandShapeOp::create(
          rewriter, loc,
          ArrayRef<int64_t>{in.getShape()[0], in.getShape()[1], 1, 1}, input,
          ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}});
      output = memref::ExpandShapeOp::create(
          rewriter, loc, ArrayRef<int64_t>{out.getShape()[0], channels, 1, 1},
          output, ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}});
    } else {
      auto steps =
          dyn_cast_or_null<DenseI64ArrayAttr>(convolution.get("stride"));
      auto dilation =
          dyn_cast_or_null<DenseI64ArrayAttr>(convolution.get("dilation"));
      auto permutation = dyn_cast_or_null<DenseI64ArrayAttr>(
          convolution.get("weight_permutation"));
      auto groups = dyn_cast_or_null<IntegerAttr>(convolution.get("groups"));
      if (in.getRank() != 4 || w.getRank() != 4 || out.getRank() != 4 ||
          !steps || steps.size() != 2 || steps[0] != steps[1] || !dilation ||
          dilation.asArrayRef() != ArrayRef<int64_t>{1, 1} || !groups ||
          groups.getInt() != 1 || !permutation ||
          (permutation.asArrayRef() != ArrayRef<int64_t>{2, 3, 1, 0} &&
           permutation.asArrayRef() != ArrayRef<int64_t>{0, 1, 2, 3}))
        return op.emitError("GemminiBall requires ordinary NCHW convolution "
                            "with HWIO weights and unit dilation");
      kh = w.getShape()[0];
      kw = w.getShape()[1];
      stride = steps[0];
      channels = out.getShape()[1];
      weight = memref::CollapseShapeOp::create(
          rewriter, loc, weight,
          ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}});
    }
    auto biasType = MemRefType::get({channels}, rewriter.getI32Type());
    Value bias = memref::AllocOp::create(rewriter, loc, biasType);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    linalg::FillOp::create(rewriter, loc, zero, bias);
    GemminiMemConvOp::create(
        rewriter, loc, input, weight, bias, output,
        rewriter.getI64IntegerAttr(kh), rewriter.getI64IntegerAttr(kw),
        rewriter.getI64IntegerAttr(stride), rewriter.getI64IntegerAttr(0),
        rewriter.getF32FloatAttr(1.0), rewriter.getF32FloatAttr(1.0));
    memref::DeallocOp::create(rewriter, loc, bias);
    if (packedWeight)
      memref::DeallocOp::create(rewriter, loc, packedWeight);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
namespace mlir::buddy {
void populateGemminiBallLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<Int8ContractionLowering>(patterns.getContext());
}
} // namespace mlir::buddy
