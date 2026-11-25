//===- ConvertLinalgToAcc.cpp - Pebble hart lowering ----------------------===//
//
// convert-linalg-to-acc recognizes the computations of the hart and fuses
// them into custom operations (FormCustomOps.cpp, FormSpatialOps.cpp), then
// lowers those to bank dataflow: the hart tiles them at bank granularity and
// connects the banks; every Ball computation is one virtual-bank operation of
// the Ball's dialect, its contract.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "fixedOps.h"
#include "im2colOps.h"
#include "int2fpOps.h"
#include "int8addOps.h"
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
#include <limits>

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

int64_t pebble::matmulRows() { return target::getTarget().bankDepth / 4; }

int64_t pebble::tileRows() {
  return target::getBallParam("SMATMUL_OS", "tileRows");
}

std::pair<Value, Value> pebble::dmaRows(OpBuilder &b, Location loc,
                                        Value byteStride, int64_t blockBytes,
                                        int64_t maximumRows) {
  int64_t span = (target::getTarget().dmaMaxRanges - 1) * 4096;
  if (span < blockBytes)
    llvm::report_fatal_error("DMA range capacity cannot hold a transfer block");
  auto index = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  Value zero = index(0), one = index(1);
  Value encodable = arith::AndIOp::create(
      b, loc,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, byteStride,
                            zero),
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, byteStride,
                            index(4095)));
  Value count = arith::AddIOp::create(
      b, loc, one,
      arith::DivUIOp::create(b, loc, index(span - blockBytes),
                             arith::MaxUIOp::create(b, loc, byteStride, one)));
  count = arith::MinUIOp::create(b, loc, count, index(maximumRows));
  // A single-row transfer does not consume a row pitch from its descriptor.
  return {
      arith::SelectOp::create(b, loc, encodable, byteStride, index(blockBytes)),
      arith::SelectOp::create(b, loc, encodable, count, one)};
}

int64_t pebble::im2colGroups(int64_t kernel, int64_t blocks, int64_t windows) {
  int64_t depth = target::getTarget().bankDepth;
  int64_t rows = (windows + tileRows() - 1) / tileRows() * tileRows();
  int64_t tiles = kernel * kernel * blocks;
  // A group holds rows x 16 tile lines of A and 16 lines per tile of B.
  int64_t perGroup = std::min(depth / rows, depth / 16);
  for (int64_t groups = 1; groups <= (target::getTarget().bankNum - 2) / 2;
       ++groups)
    if (tiles % groups == 0 && tiles / groups <= perGroup)
      return groups;
  return 0;
}

int64_t pebble::convChunk(int64_t blocks, bool depthwise) {
  if (depthwise)
    return 1;
  int64_t chunk = 8;
  while (blocks % chunk)
    --chunk;
  return chunk;
}

pebble::ConvTiling pebble::convTiling(int64_t height, int64_t width,
                                      int64_t pool, int64_t stride,
                                      int64_t kernel, int64_t blocks,
                                      bool depthwise, bool floating) {
  int64_t depth = target::getTarget().bankDepth;
  int64_t maxSide = target::getBallParam("IM2COL", "maxIter");
  int64_t largest = std::max(height, width) * pool;
  ConvTiling best{0, 0, 0};
  int64_t bestCost = std::numeric_limits<int64_t>::max();
  for (int64_t side = largest; side >= pool; side -= 1) {
    if (side % pool ||
        (side * side + tileRows() - 1) / tileRows() * tileRows() > matmulRows())
      continue;
    int64_t inputSide = (side - 1) * stride + kernel;
    if (inputSide > maxSide ||
        inputSide * inputSide * (floating ? 4 : 1) > depth)
      continue;
    int64_t chunk = convChunk(blocks, depthwise);
    int64_t groups = im2colGroups(kernel, chunk, side * side);
    // Additional K chunks keep the preceding accumulator alive.
    if (!groups ||
        (chunk != blocks && 2 * groups + 3 > target::getTarget().bankNum - 1))
      continue;
    // Count padded array rows and the word-input conversion (or output
    // processing) across the image, with one array tile of setup per tile.
    // Oversized edge tiles repeat arithmetic; tiny tiles repeat input halos.
    int64_t tiles = ((height * pool + side - 1) / side) *
                    ((width * pool + side - 1) / side);
    int64_t rows = (side * side + tileRows() - 1) / tileRows() * tileRows();
    int64_t processed = floating ? inputSide * inputSide : side * side;
    int64_t cost = tiles * (rows + processed + tileRows());
    if (cost < bestCost) {
      bestCost = cost;
      best = {side, inputSide, chunk};
    }
  }
  return best;
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

Location pebble::fusedLoc(Location result, Location quantization) {
  MLIRContext *context = result.getContext();
  return FusedLoc::get(
      context,
      {result, NameLoc::get(StringAttr::get(context, "input_quantization"),
                            quantization)});
}

// A custom operation given as input, not formed from linalg, has no finer
// origin than its own location.
Location pebble::quantizationLoc(Operation *op) {
  auto fused = dyn_cast<FusedLoc>(op->getLoc());
  if (!fused)
    return op->getLoc();
  for (Location loc : fused.getLocations())
    if (auto name = dyn_cast<NameLoc>(loc))
      if (name.getName() == "input_quantization")
        return name.getChildLoc();
  llvm::report_fatal_error("fused computation without input quantization");
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
    return "Recognize the hart computations and lower them to the hart bank "
           "dataflow";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AccDialect, ::buddy::fixed_ball::FixedBallDialect,
                    ::buddy::im2col_ball::Im2colBallDialect,
                    ::buddy::int2fp_ball::Int2FpBallDialect,
                    ::buddy::int8mul_ball::Int8MulBallDialect,
                    ::buddy::int8add_ball::Int8AddBallDialect,
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
    if (failed(pebble::lowerTopK(getOperation())))
      return signalPassFailure();
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
      if (kind == "maxpool_i32")
        status = pebble::lowerMaxPool(op, result);
      else if (kind == "pool_quant_i8" || kind == "maxpool_i8")
        status = pebble::lowerPool(op, result);
      else if (kind == "global_avg_i8")
        status = pebble::lowerGlobalAverage(op, result);
      else if (kind == "lookup_i8" || kind == "gate_mul_i8")
        status = pebble::lowerChannelOp(op, result);
      else if (kind == "transpose_i8")
        status = pebble::lowerTranspose(op, result);
      else if (kind == "batch_matmul_i8")
        status = pebble::lowerBatchMatmul(op, result);
      else if (kind == "linear_i8" || kind == "linear_i8_requant" ||
               kind == "matmul_i8")
        status = pebble::lowerLinear(
            op, residentVectors.lookup(op.getInputs()[0]), result, resident);
      else
        op.emitOpError("has no bank lowering on this hart for ") << kind;
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
    if (failed(pebble::lowerQuantize(getOperation())))
      return signalPassFailure();
    if (failed(pebble::lowerFixedOps(getOperation())))
      return signalPassFailure();
    if (failed(pebble::lowerNonlinearOps(getOperation())))
      return signalPassFailure();
  }
};

PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
