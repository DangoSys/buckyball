//===- LowerQuantize.cpp - Streaming FP32 to INT8 conversion -------------===//
#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "toint8Ops.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult pebble::lowerQuantize(func::FuncOp function) {
  SmallVector<::buddy::frontend::FP32ToInt8Op> operations;
  function.walk(
      [&](::buddy::frontend::FP32ToInt8Op op) { operations.push_back(op); });
  for (auto op : operations) {
    auto type = cast<RankedTensorType>(op.getInput().getType());
    auto factors = cast<RankedTensorType>(op.getMultiplier().getType());
    int64_t rank = type.getRank();
    if (!type.hasStaticShape() || rank < 2)
      return op.emitOpError("requires a static tensor of rank at least two");
    int64_t channels = type.getDimSize(rank - 1);
    // The fixed-point lowering owns explicitly packed periodic constants.
    // Leave that specialization intact; ordinary coefficients use this path.
    Value packed = op.getMultiplier();
    if (auto slice = packed.getDefiningOp<tensor::ExtractSliceOp>()) {
      if (llvm::all_of(slice.getStaticOffsets(),
                       [](int64_t n) { return n == 0; }) &&
          llvm::all_of(slice.getStaticStrides(),
                       [](int64_t n) { return n == 1; }))
        packed = slice.getSource();
    }
    if (auto constant = packed.getDefiningOp<arith::ConstantOp>()) {
      auto coefficients = dyn_cast<DenseFPElementsAttr>(constant.getValue());
      int64_t period = std::lcm(channels, int64_t(16));
      if (coefficients && (rank == 2 || rank == 4) &&
          coefficients.getNumElements() == period && period > channels &&
          target::getTarget().bankDepth * period <=
              (target::getTarget().dmaMaxRanges - 1) * 4096) {
        SmallVector<APFloat> values(coefficients.getValues<APFloat>());
        bool periodic = true;
        for (int64_t i = channels; i < period; ++i)
          periodic &= values[i].bitwiseIsEqual(values[i % channels]);
        if (periodic)
          continue;
      }
    }
    if (factors.getNumElements() != 1 &&
        (factors.getRank() != rank ||
         factors.getDimSize(rank - 1) != channels ||
         !llvm::all_of(factors.getShape().drop_back(),
                       [](int64_t n) { return n == 1; })))
      return op.emitOpError("requires scalar or last-axis channel multipliers");
    int64_t rows = type.getDimSize(rank - 2);
    Value storage = op.getInput();
    while (auto slice = storage.getDefiningOp<tensor::ExtractSliceOp>())
      storage = slice.getSource();
    auto storageType = cast<RankedTensorType>(storage.getType());
    int64_t storageChannels = storageType.getShape().back();
    int64_t capacity = std::min(target::getTarget().bankDepth / 4,
                                (target::getTarget().dmaMaxRanges - 1) * 4096 /
                                    (storageChannels * 4));
    if (capacity <= 0)
      return op.emitOpError(
          "channel stride exceeds the prepared DMA range capacity");
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](OpBuilder &b, int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    Value zero = index(b, 0);
    SmallVector<int64_t> strides(rank, ShapedType::kDynamic);
    strides.back() = 1;
    Value input = bufferization::ToBufferOp::create(
        b, loc,
        MemRefType::get(type.getShape(), b.getF32Type(),
                        StridedLayoutAttr::get(b.getContext(),
                                               ShapedType::kDynamic, strides)),
        op.getInput(), true);
    Value inputStride = arith::MulIOp::create(
        b, loc,
        memref::ExtractStridedMetadataOp::create(b, loc, input)
            .getStrides()[rank - 2],
        index(b, 4));
    Value output = memref::AllocOp::create(
        b, loc, MemRefType::get(type.getShape(), b.getI8Type()));
    Value multiplier =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getF32Type()));
    auto channelBlock = [&](OpBuilder &b, Value channel, int64_t lanes) {
      for (int64_t lane = 0; lane < 16; ++lane) {
        SmallVector<Value> at(factors.getRank(), zero);
        if (factors.getNumElements() != 1)
          at.back() = arith::AddIOp::create(
              b, loc, channel, index(b, std::min(lane, lanes - 1)));
        Value factor =
            tensor::ExtractOp::create(b, loc, op.getMultiplier(), at);
        memref::StoreOp::create(b, loc, factor, multiplier,
                                ValueRange{index(b, lane)});
      }
      Value scales = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      scales =
          BankMvinOp::create(b, loc, b.getI64Type(), multiplier, scales, 4, 1);
      {
        OpBuilder::InsertionGuard guard(b);
        SmallVector<OpFoldResult> prefix;
        for (int64_t axis = 0; axis < rank - 2; ++axis) {
          auto loop = scf::ForOp::create(
              b, loc, zero, index(b, type.getDimSize(axis)), index(b, 1));
          b.setInsertionPointToStart(loop.getBody());
          prefix.push_back(loop.getInductionVar());
        }
        auto chunk = [&](OpBuilder &b, Value row, int64_t count) {
          Value source = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          if (lanes != 16)
            source = BankClearOp::create(b, loc, b.getI64Type(), source, 1, 1);
          int64_t block = lanes == 16 ? 4 : 1;
          for (int64_t quarter = 0; quarter < (lanes + 3) / 4;
               quarter += block) {
            SmallVector<OpFoldResult> at(prefix);
            at.push_back(row);
            at.push_back(
                arith::AddIOp::create(b, loc, channel, index(b, quarter * 4))
                    .getResult());
            Value view = memref::SubViewOp::create(
                b, loc, input, at,
                SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
                SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
            source = BankMvinStridedOp::create(
                b, loc, b.getI64Type(), view, source, index(b, quarter),
                index(b, count * block), inputStride, 0, 4,
                index(b, std::min<int64_t>(4, lanes - quarter * 4) * 4),
                index(b, block == 4 ? 2 : 0));
          }
          Value destination = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          destination = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
              b, loc, b.getI64Type(), source, scales, destination, count * 4);
          SmallVector<OpFoldResult> at(prefix);
          at.append({row, channel});
          Value view = memref::SubViewOp::create(
              b, loc, output, at,
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
          BankMvoutStridedOp::create(b, loc, b.getI64Type(), view, destination,
                                     zero, index(b, count), index(b, channels),
                                     0, 1, index(b, lanes), zero);
          BankReleaseOp::create(b, loc, source);
          BankReleaseOp::create(b, loc, destination);
        };
        int64_t full = rows / capacity * capacity;
        repeat(b, loc, 0, full, capacity, false, {},
               [&](OpBuilder &b, Value row, ValueRange) {
                 chunk(b, row, capacity);
                 return SmallVector<Value>{};
               });
        if (rows != full)
          chunk(b, index(b, full), rows - full);
      }
      BankReleaseOp::create(b, loc, scales);
    };
    int64_t fullChannels = channels / 16 * 16;
    repeat(b, loc, 0, fullChannels, 16, false, {},
           [&](OpBuilder &b, Value channel, ValueRange) {
             channelBlock(b, channel, 16);
             return SmallVector<Value>{};
           });
    if (fullChannels != channels)
      channelBlock(b, index(b, fullChannels), channels - fullChannels);
    Value result = bufferization::ToTensorOp::create(
        b, loc, op.getResult().getType(), output, true, true);
    op.replaceAllUsesWith(result);
    op.erase();
  }
  return success();
}
