//===- ConvertLinalgToAcc.cpp - Pebble core lowering ----------------------===//
//
// convert-linalg-to-acc recognizes the computations of the core and fuses
// them into custom operations (FormCustomOps.cpp, FormSpatialOps.cpp), then
// lowers those to bank dataflow: the core tiles them at bank granularity and
// connects the banks; every Ball computation is one virtual-bank operation of
// the Ball's dialect, its contract.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "im2colOps.h"
#include "int2fpOps.h"
#include "int8mulOps.h"
#include "lutOps.h"
#include "mataddOps.h"
#include "maxpoolOps.h"
#include "smatmulOps.h"
#include "toint8Ops.h"
#include "transposeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::buddy::acc;

// The recipe supplies contiguous tensors; offsets may point into packed
// weights.
Value pebble::tensorBuffer(OpBuilder &builder, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> strides(type.getRank());
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = stride;
    stride *= type.getDimSize(dim);
  }
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, strides);
  auto memory = MemRefType::get(type.getShape(), type.getElementType(), layout);
  return bufferization::ToBufferOp::create(builder, loc, memory, tensor, true);
}

SmallVector<Value> pebble::repeat(
    OpBuilder &builder, Location loc, int64_t begin, int64_t end, int64_t step,
    bool unroll, ValueRange carried,
    function_ref<SmallVector<Value>(OpBuilder &, Value, ValueRange)> body) {
  SmallVector<Value> values(carried);
  if (unroll || end - begin <= step) {
    for (int64_t index = begin; index < end; index += step)
      values = body(
          builder, arith::ConstantIndexOp::create(builder, loc, index), values);
    return values;
  }
  auto loop = scf::ForOp::create(
      builder, loc, arith::ConstantIndexOp::create(builder, loc, begin),
      arith::ConstantIndexOp::create(builder, loc, end),
      arith::ConstantIndexOp::create(builder, loc, step), carried,
      [&](OpBuilder &nested, Location loc, Value index, ValueRange args) {
        scf::YieldOp::create(nested, loc, body(nested, index, args));
      });
  return loop.getResults();
}

int64_t pebble::matmulRows() {
  return buckyball_target::getBuckyballTarget().bankDepth / 4;
}

int64_t pebble::tileRows() {
  return buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
}

int64_t pebble::im2colGroups(int64_t kernel, int64_t blocks, int64_t windows) {
  int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
  int64_t rows = (windows + tileRows() - 1) / tileRows() * tileRows();
  int64_t tiles = kernel * kernel * blocks;
  // A group holds rows x 16 tile lines of A and 16 lines per tile of B.
  int64_t perGroup = std::min(depth / rows, depth / 16);
  for (int64_t groups = 1; groups <= 9; ++groups)
    if (tiles % groups == 0 && tiles / groups <= perGroup)
      return groups;
  return 0;
}

pebble::ConvTiling pebble::convTiling(int64_t height, int64_t width,
                                      int64_t pool, int64_t stride,
                                      int64_t kernel, int64_t blocks,
                                      bool depthwise, bool floating) {
  int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
  int64_t maxSide =
      buckyball_target::getBuckyballBallParam("IM2COL", "maxIter");
  int64_t largest = std::max(height, width) * pool;
  for (int64_t side = largest; side >= pool; side -= 1) {
    if (side % pool ||
        (side * side + tileRows() - 1) / tileRows() * tileRows() > matmulRows())
      continue;
    int64_t inputSide = (side - 1) * stride + kernel;
    if (inputSide > maxSide ||
        inputSide * inputSide * (floating ? 4 : 1) > depth)
      continue;
    int64_t limit = depthwise ? 1 : std::min<int64_t>(blocks, 8);
    for (int64_t chunk = limit; chunk >= 1; --chunk)
      if (im2colGroups(kernel, chunk, side * side))
        return {side, inputSide, chunk};
  }
  return {0, 0, 0};
}

void pebble::observeResident(Operation *op, Value bank) {
  for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
    if (!isa<::buddy::frontend::TraceEndOp>(user))
      continue;
    OpBuilder builder(user);
    Operation *trace = builder.clone(*user);
    trace->setOperand(0, bank);
    trace->getResult(0).setType(bank.getType());
    user->erase();
  }
}

void pebble::moveTraceStarts(Operation *producer, ArrayRef<Operation *> ends) {
  for (Operation *end : ends)
    for (Operation *op = end->getPrevNode(); op != producer;
         op = op->getPrevNode()) {
      if (isa<::buddy::frontend::TraceStartOp>(op) &&
          op->getAttr("id") == end->getAttr("id") &&
          op->getAttr("id_path") == end->getAttr("id_path")) {
        op->moveBefore(producer);
        break;
      }
    }
}

namespace {
class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Recognize the core computations and lower them to the core bank "
           "dataflow";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AccDialect, ::buddy::im2col_ball::Im2colBallDialect,
                    ::buddy::int2fp_ball::Int2FpBallDialect,
                    ::buddy::int8mul_ball::Int8MulBallDialect,
                    ::buddy::lut_ball::LutBallDialect,
                    ::buddy::matadd_ball::MatAddBallDialect,
                    ::buddy::maxpool_ball::MaxPoolBallDialect,
                    ::buddy::smatmul_ball::SMatMulBallDialect,
                    ::buddy::toint8_ball::ToInt8BallDialect,
                    ::buddy::transpose_ball::TransposeBallDialect,
                    arith::ArithDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    scf::SCFDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    pebble::formCustomOps(getOperation());
    DenseMap<Value, Value> residentVectors;
    SmallVector<Operation *> consumed;
    if (failed(
            pebble::lowerSpatialOps(getOperation(), residentVectors, consumed)))
      return signalPassFailure();
    SmallVector<::buddy::acc::CustomOp> computations;
    getOperation().walk([&](::buddy::acc::CustomOp op) {
      if (!llvm::is_contained(consumed, op.getOperation()))
        computations.push_back(op);
    });
    for (auto op : computations) {
      StringRef kind = op.getKind();
      Value result, resident;
      LogicalResult status = failure();
      if (kind == "global_avg_i8")
        status = pebble::lowerGlobalAverage(op, result);
      else if (kind == "lookup_i8" || kind == "gate_mul_i8")
        status = pebble::lowerChannelOp(op, result);
      else if (kind == "transpose_i8")
        status = pebble::lowerTranspose(op, result);
      else if (kind == "linear_i8" || kind == "linear_i8_requant" ||
               kind == "matmul_i8")
        status = pebble::lowerLinear(
            op, residentVectors.lookup(op.getInputs()[0]), result, resident);
      else
        op.emitOpError("has no bank lowering on this core for ") << kind;
      if (failed(status))
        return signalPassFailure();
      if (resident) {
        residentVectors[op.getResult(0)] = resident;
        pebble::observeResident(op, resident);
        consumed.push_back(op);
        continue;
      }
      op.getResult(0).replaceAllUsesWith(result);
      op.erase();
    }
    for (Operation *op : llvm::reverse(consumed))
      op->erase();
  }
};

PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
