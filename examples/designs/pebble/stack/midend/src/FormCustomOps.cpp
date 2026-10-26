//===- FormCustomOps.cpp - Recognize and fuse core computations -----------===//

#include "Custom/CustomOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using ::buddy::custom::CustomOp;

namespace {
Operation *frontend(Value value, StringRef name) {
  Operation *op = value.getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}

} // namespace

bool pebble::elementwise(linalg::GenericOp op, StringRef name) {
  if (!op || !op.hasPureTensorSemantics() ||
      !linalg::isaElemwiseSingleBinaryOpInterface(op) ||
      op.getBody()->front().getName().getStringRef() != name)
    return false;
  auto result = cast<RankedTensorType>(op.getResult(0).getType());
  auto maps = op.getIndexingMapsArray();
  if (!result.hasStaticShape() || !maps.back().isIdentity())
    return false;
  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    auto type = cast<RankedTensorType>(input.getType());
    if (!type.hasStaticShape() || type.getRank() != result.getRank())
      return false;
    SmallVector<AffineExpr> dimensions;
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      dimensions.push_back(type.getDimSize(dim) == 1 &&
                                   result.getDimSize(dim) != 1
                               ? getAffineConstantExpr(0, op.getContext())
                               : getAffineDimExpr(dim, op.getContext()));
    if (maps[index] !=
        AffineMap::get(type.getRank(), 0, dimensions, op.getContext()))
      return false;
  }
  return true;
}

namespace {
class FormCustomOpsPass
    : public PassWrapper<FormCustomOpsPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormCustomOpsPass)
  StringRef getArgument() const final { return "form-custom-ops"; }
  StringRef getDescription() const final {
    return "Recognize core computations and fuse them into custom tensor ops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::custom::CustomDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    // MLIR checks both indexing maps and scalar computations, including casts.
    SmallVector<linalg::GenericOp> generics;
    getOperation().walk([&](linalg::GenericOp op) { generics.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (auto op : generics) {
      if (op.hasPureTensorSemantics() &&
          (linalg::isaContractionOpInterface(op) ||
           linalg::isaTransposeOpInterface(op))) {
        rewriter.setInsertionPoint(op);
        (void)linalg::specializeGenericOp(rewriter, op);
      }
    }

    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    DominanceInfo dominance(getOperation());
    for (auto op : matmuls) {
      if (!op.hasPureTensorSemantics() || op.hasUserDefinedMaps() ||
          op.getCast() != linalg::TypeFn::cast_signed)
        continue;
      rewriter.setInsertionPoint(op);
      auto a = cast<RankedTensorType>(op.getInputs()[0].getType());
      auto b = cast<RankedTensorType>(op.getInputs()[1].getType());
      auto result = cast<RankedTensorType>(op.getResult(0).getType());
      if (a.getElementType().isInteger(8) && b.getElementType().isInteger(8) &&
          result.getElementType().isInteger(32)) {
        // matmul_i8(A, B, C) = C + A @ B, with signed INT8 operands.
        auto custom = CustomOp::create(
            rewriter, op.getLoc(), TypeRange{result}, "matmul_i8",
            ValueRange{op.getInputs()[0], op.getInputs()[1],
                       op.getOutputs()[0]});
        rewriter.replaceOp(op, custom.getOutputs());
        continue;
      }

      Operation *lhs = frontend(op.getInputs()[0], "frontend.int8_to_fp32");
      Operation *rhs = frontend(op.getInputs()[1], "frontend.int8_to_fp32");
      if (!lhs || !rhs || !matchPattern(op.getOutputs()[0], m_AnyZeroFloat()))
        continue;
      Operation *quantize =
          frontend(lhs->getOperand(0), "frontend.fp32_to_int8");
      if (!quantize)
        continue;
      // Match bias dequantized with exactly input_step * weight_step.
      Value biasedResult = op.getResult(0);
      Operation *trace = nullptr;
      if (biasedResult.hasOneUse()) {
        Operation *user = *biasedResult.getUsers().begin();
        auto tensorTrace = user->getAttrOfType<BoolAttr>("tensor");
        // Cycle markers follow the logical Linear. Tensor traces keep their
        // observed boundary and therefore prevent folding bias across it.
        if (user->getName().getStringRef() == "buddy_trace.end" &&
            tensorTrace && !tensorTrace.getValue()) {
          trace = user;
          biasedResult = user->getResult(0);
        }
      }
      Value bias;
      linalg::GenericOp biasAdd, biasScale;
      Operation *biasDequant = nullptr;
      if (biasedResult.hasOneUse()) {
        auto add =
            dyn_cast<linalg::GenericOp>(*biasedResult.getUsers().begin());
        if (pebble::elementwise(add, "arith.addf") &&
            llvm::is_contained(add.getInputs(), biasedResult)) {
          Value other =
              add.getInputs()[add.getInputs()[0] == biasedResult ? 1 : 0];
          auto dequant = frontend(other, "frontend.int32_to_fp32");
          if (dequant) {
            auto scale =
                dequant->getOperand(1).getDefiningOp<linalg::GenericOp>();
            auto biasType =
                cast<RankedTensorType>(dequant->getOperand(0).getType());
            if (pebble::elementwise(scale, "arith.mulf") &&
                biasType.getElementType().isInteger(32) &&
                biasType.getShape() == result.getShape() &&
                ((scale.getInputs()[0] == lhs->getOperand(1) &&
                  scale.getInputs()[1] == rhs->getOperand(1)) ||
                 (scale.getInputs()[1] == lhs->getOperand(1) &&
                  scale.getInputs()[0] == rhs->getOperand(1))) &&
                dominance.properlyDominates(dequant->getOperand(0), op)) {
              bias = dequant->getOperand(0);
              biasAdd = add;
              biasScale = scale;
              biasDequant = dequant;
            }
          }
        }
      }
      if (!bias) {
        auto biasType =
            RankedTensorType::get(result.getShape(), rewriter.getI32Type());
        bias = arith::ConstantOp::create(rewriter, op.getLoc(), biasType,
                                         rewriter.getZeroAttr(biasType));
      }
      // linear_i8(x, W, bias, multiplier, input_step, weight_step) accumulates
      // signed INT8 products on INT32 bias, then restores FP32 values.
      auto custom = CustomOp::create(
          rewriter, op.getLoc(), TypeRange{result}, "linear_i8",
          ValueRange{quantize->getOperand(0), rhs->getOperand(0), bias,
                     quantize->getOperand(1), lhs->getOperand(1),
                     rhs->getOperand(1)});
      rewriter.replaceOp(op, custom.getOutputs());
      if (biasAdd) {
        rewriter.replaceOp(biasAdd,
                           trace ? trace->getResult(0) : custom.getResult(0));
        if (biasDequant->use_empty())
          rewriter.eraseOp(biasDequant);
        if (biasScale->use_empty())
          rewriter.eraseOp(biasScale);
      }
      if (rhs != lhs && rhs->use_empty())
        rewriter.eraseOp(rhs);
      if (lhs->use_empty())
        rewriter.eraseOp(lhs);
      if (quantize->use_empty())
        rewriter.eraseOp(quantize);
    }

    SmallVector<linalg::TransposeOp> transposes;
    getOperation().walk(
        [&](linalg::TransposeOp op) { transposes.push_back(op); });
    const auto &target = buckyball_target::getBuckyballTarget();
    for (auto op : transposes) {
      // TRANSPOSE holds the whole matrix in two groups of bank columns of at
      // most 31 banks each.
      auto type = dyn_cast<RankedTensorType>(op.getInput().getType());
      if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
          !type.getElementType().isInteger(8) ||
          op.getPermutation() != ArrayRef<int64_t>({1, 0}) ||
          type.getDimSize(0) > target.bankDepth || type.getDimSize(1) % 16 ||
          type.getDimSize(1) / 16 > 31 ||
          2 * type.getDimSize(1) / 16 > target.bankNum)
        continue;
      rewriter.setInsertionPoint(op);
      auto custom =
          CustomOp::create(rewriter, op.getLoc(), op->getResultTypes(),
                           "transpose_i8", ValueRange{op.getInput()});
      rewriter.replaceOp(op, custom.getOutputs());
    }
    pebble::formSpatialOps(getOperation());
  }
};

PassRegistration<FormCustomOpsPass> registration;
} // namespace
