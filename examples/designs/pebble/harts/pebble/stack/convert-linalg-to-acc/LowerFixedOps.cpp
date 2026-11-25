#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "fixedOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "toint8Ops.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

namespace {
LogicalResult lowerMean(FixedMeanOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(op.getInput().getType());
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  if (!inputType.hasStaticShape() || inputType.getRank() != 4 ||
      inputType.getDimSize(1) <= 0 || inputType.getDimSize(2) <= 0 ||
      resultType.getShape() != ArrayRef<int64_t>{inputType.getDimSize(0), 1, 1,
                                                 inputType.getDimSize(3)})
    return op.emitOpError("requires a nonempty static NHWC spatial mean");
  int64_t channels = inputType.getDimSize(3);
  int64_t height = inputType.getDimSize(1), width = inputType.getDimSize(2);
  int64_t pixels = std::min(width, target::getTarget().bankDepth / 4);
  auto index = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  Value zero = index(0), one = index(1);
  Value input = bufferization::ToBufferOp::create(
      b, loc,
      MemRefType::get(inputType.getShape(), b.getI32Type(),
                      StridedLayoutAttr::get(
                          b.getContext(), ShapedType::kDynamic,
                          SmallVector<int64_t>(4, ShapedType::kDynamic))),
      op.getInput(), true);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(resultType.getShape(), b.getI32Type()));
  Value divisors = arith::ConstantOp::create(
      b, loc,
      DenseIntElementsAttr::get(RankedTensorType::get({16}, b.getI32Type()),
                                SmallVector<int32_t>(16, height * width)));
  divisors = pebble::tensorBuffer(b, loc, divisors);
  Value stride = arith::MulIOp::create(
      b, loc,
      memref::ExtractStridedMetadataOp::create(b, loc, input).getStrides()[2],
      index(4));
  auto batchLoop =
      scf::ForOp::create(b, loc, zero, index(inputType.getDimSize(0)), one);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(batchLoop.getBody());
    Value n = batchLoop.getInductionVar();
    auto channelBlock = [&](Value channel, int64_t lanes) {
      Value sum = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      sum = BankClearOp::create(b, loc, b.getI64Type(), sum, 1, 1);
      Value data = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      if (lanes < 16 && width % pixels == 0)
        data = BankClearOp::create(b, loc, b.getI64Type(), data, 1, 1);
      auto rows = scf::ForOp::create(b, loc, zero, index(height), one,
                                     ValueRange{sum, data});
      b.setInsertionPointToStart(rows.getBody());
      Value y = rows.getInductionVar();
      auto columns = scf::ForOp::create(
          b, loc, zero, index(width), index(pixels), rows.getRegionIterArgs());
      b.setInsertionPointToStart(columns.getBody());
      Value x = columns.getInductionVar();
      Value count = arith::MinSIOp::create(
          b, loc, index(pixels),
          arith::SubIOp::create(b, loc, index(width), x));
      data = columns.getRegionIterArgs()[1];
      if (width % pixels)
        data = BankClearOp::create(b, loc, b.getI64Type(), data, 1, 1);
      if (lanes == 8 || lanes == 16) {
        Value view = memref::SubViewOp::create(
            b, loc, input, ArrayRef<OpFoldResult>{n, y, x, channel},
            SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
            SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
        data = BankMvinStridedOp::create(
            b, loc, b.getI64Type(), view, data, zero,
            arith::MulIOp::create(b, loc, count, index(lanes / 4)), stride, 0,
            4, index(16), index(lanes == 16 ? 2 : 1));
      } else {
        for (int64_t quarter = 0; quarter < (lanes + 3) / 4; ++quarter) {
          Value c = arith::AddIOp::create(b, loc, channel, index(quarter * 4));
          Value view = memref::SubViewOp::create(
              b, loc, input, ArrayRef<OpFoldResult>{n, y, x, c},
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
          data = BankMvinStridedOp::create(
              b, loc, b.getI64Type(), view, data, index(quarter), count, stride,
              0, 4, index(std::min<int64_t>(4, lanes - quarter * 4) * 4),
              arith::ConstantIndexOp::create(b, loc, 0));
        }
      }
      Value next = ::buddy::fixed_ball::BankFixedOp::create(
          b, loc, b.getI64Type(), data, columns.getRegionIterArgs()[0],
          columns.getRegionIterArgs()[0], pixels * 4, 6, 24, false, false);
      scf::YieldOp::create(b, loc, ValueRange{next, data});
      b.setInsertionPointAfter(columns);
      scf::YieldOp::create(b, loc, columns.getResults());
      b.setInsertionPointAfter(rows);
      BankReleaseOp::create(b, loc, rows.getResult(1));
      Value divisorBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      divisorBank = BankMvinOp::create(b, loc, b.getI64Type(), divisors,
                                       divisorBank, 4, 1);
      Value resultBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      resultBank = ::buddy::fixed_ball::BankFixedOp::create(
          b, loc, b.getI64Type(), rows.getResult(0), divisorBank, resultBank, 4,
          7, 24, false, false);
      BankReleaseOp::create(b, loc, rows.getResult(0));
      BankReleaseOp::create(b, loc, divisorBank);
      if (lanes == 8 || lanes == 16) {
        Value view = memref::SubViewOp::create(
            b, loc, output, ArrayRef<OpFoldResult>{n, zero, zero, channel},
            SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
            SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
        resultBank = BankMvoutStridedOp::create(
            b, loc, b.getI64Type(), view, resultBank, zero, index(lanes / 4),
            index(16), 0, 1, index(16), zero);
      } else {
        for (int64_t quarter = 0; quarter < (lanes + 3) / 4; ++quarter) {
          Value c = arith::AddIOp::create(b, loc, channel, index(quarter * 4));
          Value view = memref::SubViewOp::create(
              b, loc, output, ArrayRef<OpFoldResult>{n, zero, zero, c},
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
          resultBank = BankMvoutStridedOp::create(
              b, loc, b.getI64Type(), view, resultBank, index(quarter), one,
              index(16), 0, 1,
              index(std::min<int64_t>(4, lanes - quarter * 4) * 4),
              arith::ConstantIndexOp::create(b, loc, 0));
        }
      }
      BankReleaseOp::create(b, loc, resultBank);
    };
    if (channels >= 16) {
      auto channelsLoop = scf::ForOp::create(
          b, loc, zero, index(channels / 16 * 16), index(16));
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(channelsLoop.getBody());
      channelBlock(channelsLoop.getInductionVar(), 16);
    }
    if (channels % 16)
      channelBlock(index(channels / 16 * 16), channels % 16);
  }
  Value result =
      bufferization::ToTensorOp::create(b, loc, resultType, output, true, true);
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}
} // namespace

LogicalResult pebble::lowerFixedOps(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *op) {
    if (isa<FixedRequantOp, FixedUnaryOp, FixedBinaryOp, FixedMeanOp,
            Int32ToInt8Op, FP32ToInt8Op>(op))
      operations.push_back(op);
  });
  for (Operation *op : operations) {
    if (auto mean = dyn_cast<FixedMeanOp>(op)) {
      if (failed(lowerMean(mean)))
        return failure();
      continue;
    }
    OpBuilder b(op);
    Location loc = op->getLoc();
    Operation *quant = isa<Int32ToInt8Op, FP32ToInt8Op>(op) ? op : nullptr;
    bool floatingQuant = isa<FP32ToInt8Op>(op);
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t rank = type.getRank();
    if (!type.hasStaticShape() || (rank != 2 && rank != 4) ||
        inputType.getShape() != type.getShape() ||
        !(inputType.getElementType().isInteger(32) ||
          (floatingQuant && inputType.getElementType().isF32())))
      return op->emitOpError(
          "requires static rank two or NHWC fixed-point tensors");
    int64_t channels = type.getDimSize(rank - 1);
    int64_t width = type.getDimSize(rank - 2);
    int64_t pixels = std::min(width, target::getTarget().bankDepth / 4);
    auto requant = dyn_cast<FixedRequantOp>(op);
    auto unary = dyn_cast<FixedUnaryOp>(op);
    auto binary = dyn_cast<FixedBinaryOp>(op);
    int64_t storageChannels = channels;
    for (unsigned i = 0; i < (binary ? 2u : 1u); ++i) {
      Value storage = op->getOperand(i);
      while (auto slice = storage.getDefiningOp<tensor::ExtractSliceOp>())
        storage = slice.getSource();
      storageChannels =
          std::max(storageChannels,
                   cast<RankedTensorType>(storage.getType()).getShape().back());
    }
    int64_t dmaSpan = (target::getTarget().dmaMaxRanges - 1) * 4096;
    // One pixel transfers at most sixteen words; only subsequent pixels
    // contribute the channel stride to the DMA address span.
    pixels = std::min(pixels, 1 + (dmaSpan - 64) / (storageChannels * 4));
    if (pixels <= 0)
      return op->emitOpError("channel stride exceeds prepared DMA capacity");
    int64_t mode = 0;
    if (unary) {
      if (unary.getKind() == "hardswish")
        mode = 1;
      else if (unary.getKind() == "hardsigmoid")
        mode = 2;
      else if (unary.getKind() == "relu")
        mode = 4;
      else
        return op->emitOpError("unknown fixed unary operation");
    }
    bool broadcast = false;
    if (binary) {
      if (binary.getKind() == "mul")
        mode = 3;
      else if (binary.getKind() == "add")
        mode = 5;
      else
        return op->emitOpError("unknown fixed binary operation");
      auto rhs = cast<RankedTensorType>(binary.getRhs().getType());
      broadcast = rhs != type;
      if (broadcast && (mode != 3 || rhs.getRank() != rank ||
                        rhs.getDimSize(rank - 1) != channels ||
                        !llvm::all_of(rhs.getShape().drop_back(),
                                      [](int64_t n) { return n == 1; })))
        return op->emitOpError(
            "supports full tensors or per-channel multiplication");
    }
    auto index = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    Value zero = index(0), one = index(1);
    auto buffer = [&](Value tensor) -> Value {
      auto t = cast<RankedTensorType>(tensor.getType());
      return bufferization::ToBufferOp::create(
          b, loc,
          MemRefType::get(
              t.getShape(), t.getElementType(),
              StridedLayoutAttr::get(
                  b.getContext(), ShapedType::kDynamic,
                  SmallVector<int64_t>(t.getRank(), ShapedType::kDynamic))),
          tensor, true);
    };
    Value input = buffer(op->getOperand(0));
    Value rhs;
    if (binary)
      rhs = buffer(binary.getRhs());
    Value multipliers, shifts;
    if (quant) {
      Value parameter = op->getOperand(1);
      if (auto slice = parameter.getDefiningOp<tensor::ExtractSliceOp>()) {
        if (!llvm::all_of(slice.getStaticOffsets(),
                          [](int64_t n) { return n == 0; }) ||
            !llvm::all_of(slice.getStaticStrides(),
                          [](int64_t n) { return n == 1; }))
          return op->emitOpError(
              "requires a zero-offset multiplier channel view");
        parameter = slice.getSource();
      }
      auto parameterType = cast<RankedTensorType>(parameter.getType());
      if (parameterType.getNumElements() != (channels + 15) / 16 * 16 &&
          parameterType.getNumElements() != std::lcm(channels, int64_t(16)))
        return op->emitOpError(
            "requires per-channel multipliers padded to sixteen channels");
      ReassociationIndices axes;
      for (int64_t axis = 0; axis < parameterType.getRank(); ++axis)
        axes.push_back(axis);
      parameter = tensor::CollapseShapeOp::create(
          b, loc, parameter, ArrayRef<ReassociationIndices>{axes});
      multipliers = pebble::tensorBuffer(b, loc, parameter);
    } else if (requant) {
      multipliers = buffer(requant.getMultipliers());
      shifts = buffer(requant.getShifts());
    } else if (mode <= 2) {
      SmallVector<int32_t> identity(32, 0);
      std::fill_n(identity.begin(), 16, 1);
      Value tensor = arith::ConstantOp::create(
          b, loc,
          DenseIntElementsAttr::get(RankedTensorType::get({32}, b.getI32Type()),
                                    identity));
      multipliers = pebble::tensorBuffer(b, loc, tensor);
    }
    Value output = memref::AllocOp::create(
        b, loc, MemRefType::get(type.getShape(), type.getElementType()));
    auto contiguous = [](Value value) {
      while (true) {
        if (auto trace = value.getDefiningOp<TraceEndOp>())
          value = trace->getOperand(0);
        else if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>())
          value = collapse.getSrc();
        else if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>())
          value = expand.getSrc();
        else
          break;
      }
      if (value.getDefiningOp<arith::ConstantOp>())
        return true;
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto function =
            dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
        if (!function)
          return false;
        auto layout = function.getArgAttrOfType<StridedLayoutAttr>(
            argument.getArgNumber(), "bufferization.buffer_layout");
        if (!layout)
          return false;
        auto shape = cast<RankedTensorType>(value.getType()).getShape();
        int64_t stride = 1;
        for (int64_t axis = shape.size() - 1; axis >= 0; --axis) {
          if (layout.getStrides()[axis] != stride)
            return false;
          stride *= shape[axis];
        }
        return true;
      }
      auto tensor = value.getDefiningOp<bufferization::ToTensorOp>();
      return tensor && cast<MemRefType>(tensor.getBuffer().getType())
                           .getLayout()
                           .isIdentity();
    };
    int64_t elements = type.getNumElements();
    bool vector = elements == channels;
    bool periodicRequant =
        requant && elements > 16 && channels % 16 == 0 &&
        cast<RankedTensorType>(requant.getMultipliers().getType())
                .getNumElements() == 16 &&
        cast<RankedTensorType>(requant.getShifts().getType())
                .getNumElements() == 16;
    bool uniformQuant =
        quant &&
        cast<RankedTensorType>(op->getOperand(1).getType()).getNumElements() ==
            1;
    int64_t period = uniformQuant ? 16 : std::lcm(channels, int64_t(16));
    bool periodicQuant =
        quant && elements % period == 0 && period * 4 < 4096 &&
        cast<MemRefType>(multipliers.getType()).getNumElements() >= period &&
        (vector || contiguous(op->getOperand(0)));
    bool flatten = elements % 16 == 0 &&
                   (periodicQuant || (requant && vector) ||
                    (periodicRequant && contiguous(op->getOperand(0))) ||
                    ((unary || (binary && !broadcast)) &&
                     (vector || (contiguous(op->getOperand(0)) &&
                                 (!binary || contiguous(binary.getRhs()))))));
    if (flatten) {
      ReassociationIndices axes;
      for (int64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);
      Value flatInput = memref::CollapseShapeOp::create(
          b, loc, input, ArrayRef<ReassociationIndices>{axes});
      Value flatOutput = memref::CollapseShapeOp::create(
          b, loc, output, ArrayRef<ReassociationIndices>{axes});
      Value flatRhs;
      if (binary)
        flatRhs = memref::CollapseShapeOp::create(
            b, loc, rhs, ArrayRef<ReassociationIndices>{axes});
      int64_t capacity = target::getTarget().bankDepth /
                         (requant && !periodicRequant ? 8 : 4) * 16;
      int64_t phases = quant ? period / 16 : 1;
      if (phases > 1)
        capacity = std::min(capacity, dmaSpan / (period * 4) * 16);
      if (capacity <= 0)
        return op->emitOpError(
            "coefficient period exceeds prepared DMA capacity");
      auto phaseLoop = scf::ForOp::create(b, loc, zero, index(phases), one);
      {
        OpBuilder::InsertionGuard phaseGuard(b);
        b.setInsertionPointToStart(phaseLoop.getBody());
        Value phase = phaseLoop.getInductionVar();
        Value channel = arith::MulIOp::create(b, loc, phase, index(16));
        Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        Value other = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        Value destination = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        auto loadOther = [&](Value bank, Value first, int64_t count) -> Value {
          Value memory;
          int64_t rows = count / 4;
          if (quant) {
            memory = memref::SubViewOp::create(
                b, loc, multipliers, ArrayRef<OpFoldResult>{channel},
                ArrayRef<OpFoldResult>{b.getIndexAttr(16)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            bank =
                BankMvinOp::create(b, loc, b.getI64Type(), memory, bank, 4, 1);
          } else if (requant) {
            for (int part = 0; part < 2; ++part) {
              memory = memref::SubViewOp::create(
                  b, loc, part ? shifts : multipliers,
                  ArrayRef<OpFoldResult>{periodicRequant ? zero : first},
                  ArrayRef<OpFoldResult>{
                      b.getIndexAttr(periodicRequant ? 16 : count)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
              bank = BankMvinStridedOp::create(
                  b, loc, b.getI64Type(), memory, bank,
                  index(periodicRequant ? part * 4 : part),
                  index(periodicRequant ? 4 : rows), index(16), 0,
                  periodicRequant ? 1 : 2, index(16), zero);
            }
          } else if (binary) {
            memory = memref::SubViewOp::create(
                b, loc, flatRhs, ArrayRef<OpFoldResult>{first},
                ArrayRef<OpFoldResult>{b.getIndexAttr(count)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            bank = BankMvinOp::create(b, loc, b.getI64Type(), memory, bank,
                                      rows, 1);
          } else if (mode <= 2) {
            bank = BankMvinOp::create(b, loc, b.getI64Type(), multipliers, bank,
                                      8, 1);
          }
          return bank;
        };
        bool invariant = quant || periodicRequant || unary;
        if (invariant)
          other = loadOther(other, zero, capacity);
        int64_t length = elements / phases;
        auto loop =
            scf::ForOp::create(b, loc, zero, index(length), index(capacity),
                               ValueRange{a, other, destination});
        {
          OpBuilder::InsertionGuard loopGuard(b);
          b.setInsertionPointToStart(loop.getBody());
          Value first = loop.getInductionVar();
          Value offset = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, first, index(phases)),
              channel);
          auto tile = [&](int64_t count) -> SmallVector<Value> {
            int64_t rows = count / 4;
            Value a = loop.getRegionIterArgs()[0];
            Value other = loop.getRegionIterArgs()[1];
            Value result = loop.getRegionIterArgs()[2];
            Value memory = memref::SubViewOp::create(
                b, loc, flatInput, ArrayRef<OpFoldResult>{offset},
                ArrayRef<OpFoldResult>{b.getIndexAttr(phases == 1 ? count : 1)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            if (phases == 1)
              a = BankMvinOp::create(b, loc, b.getI64Type(), memory, a, rows,
                                     1);
            else
              a = BankMvinStridedOp::create(
                  b, loc, b.getI64Type(), memory, a, zero, index(rows),
                  index(period * 4), 0, 4, index(16), index(2));
            if (!invariant)
              other = loadOther(other, first, count);
            if (floatingQuant)
              result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
                  b, loc, b.getI64Type(), a, other, result, rows);
            else if (quant)
              result = ::buddy::toint8_ball::BankInt32ToInt8Op::create(
                  b, loc, b.getI64Type(), a, other, result, count / 16, false);
            else
              result = ::buddy::fixed_ball::BankFixedOp::create(
                  b, loc, b.getI64Type(), a, other, result, rows, mode,
                  cast<IntegerAttr>(op->getAttr("fraction")).getInt(), false,
                  bool(requant) && !periodicRequant);
            memory = memref::SubViewOp::create(
                b, loc, flatOutput, ArrayRef<OpFoldResult>{offset},
                ArrayRef<OpFoldResult>{b.getIndexAttr(phases == 1 ? count : 1)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            if (phases == 1)
              result =
                  BankMvoutOp::create(b, loc, b.getI64Type(), memory, result,
                                      quant ? count / 16 : rows, 1);
            else
              result = BankMvoutStridedOp::create(
                  b, loc, b.getI64Type(), memory, result, zero,
                  index(count / 16), index(period), 0, 1, index(16), zero);
            return {a, other, result};
          };
          int64_t tail = length % capacity;
          if (tail) {
            Value last = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                               first, index(length - tail));
            auto branch = scf::IfOp::create(
                b, loc, SmallVector<Type>(3, b.getI64Type()), last, true);
            b.setInsertionPointToStart(&branch.getThenRegion().front());
            scf::YieldOp::create(b, loc, tile(tail));
            b.setInsertionPointToStart(&branch.getElseRegion().front());
            scf::YieldOp::create(b, loc, tile(capacity));
            b.setInsertionPointAfter(branch);
            scf::YieldOp::create(b, loc, branch.getResults());
          } else {
            scf::YieldOp::create(b, loc, tile(capacity));
          }
        }
        for (Value bank : loop.getResults())
          BankReleaseOp::create(b, loc, bank);
      }
      Value tensor =
          bufferization::ToTensorOp::create(b, loc, type, output, true, true);
      op->getResult(0).replaceAllUsesWith(tensor);
      op->erase();
      continue;
    }
    auto channelLoop =
        scf::ForOp::create(b, loc, zero, index(channels), index(16));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(channelLoop.getBody());
      Value channel = channelLoop.getInductionVar();
      Value parameters = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      if (quant) {
        Value memory = memref::SubViewOp::create(
            b, loc, multipliers, ArrayRef<OpFoldResult>{channel},
            ArrayRef<OpFoldResult>{b.getIndexAttr(16)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        parameters = BankMvinOp::create(b, loc, b.getI64Type(), memory,
                                        parameters, 4, 1);
      } else if (requant) {
        Value coefficientChannel = periodicRequant ? zero : channel;
        for (int part = 0; part < 2; ++part) {
          Value memory = memref::SubViewOp::create(
              b, loc, part ? shifts : multipliers,
              ArrayRef<OpFoldResult>{coefficientChannel},
              ArrayRef<OpFoldResult>{b.getIndexAttr(16)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          parameters = BankMvinStridedOp::create(
              b, loc, b.getI64Type(), memory, parameters, index(part * 4),
              index(4), index(16), 0, 1, index(16),
              arith::ConstantIndexOp::create(b, loc, 0));
        }
      } else if (mode <= 2) {
        parameters = BankMvinOp::create(b, loc, b.getI64Type(), multipliers,
                                        parameters, 8, 1);
      } else if (broadcast) {
        // The channel gate is shared by every spatial tile of this block.
        if (channels % 16)
          parameters =
              BankClearOp::create(b, loc, b.getI64Type(), parameters, 1, 1);
        Value lanes = arith::MinSIOp::create(
            b, loc, index(16),
            arith::SubIOp::create(b, loc, index(channels), channel));
        Value quarters = arith::CeilDivUIOp::create(b, loc, lanes, index(4));
        auto load = scf::ForOp::create(b, loc, zero, quarters, one,
                                       ValueRange{parameters});
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(load.getBody());
          Value quarter = load.getInductionVar();
          Value c = arith::AddIOp::create(
              b, loc, channel,
              arith::MulIOp::create(b, loc, quarter, index(4)));
          SmallVector<OpFoldResult> address(rank, b.getIndexAttr(0));
          address.back() = c;
          Value memory = memref::SubViewOp::create(
              b, loc, rhs, address,
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
          Value bytes = arith::MulIOp::create(
              b, loc, index(4),
              arith::MinSIOp::create(
                  b, loc, index(4),
                  arith::SubIOp::create(b, loc, index(channels), c)));
          Value loaded = BankMvinStridedOp::create(
              b, loc, b.getI64Type(), memory, load.getRegionIterArgs()[0],
              quarter, one, index(16), 0, 1, bytes,
              arith::ConstantIndexOp::create(b, loc, 0));
          scf::YieldOp::create(b, loc, loaded);
        }
        parameters = load.getResult(0);
      }
      SmallVector<Value> offsets;
      for (int64_t axis = 0; axis < rank - 2; ++axis) {
        auto loop =
            scf::ForOp::create(b, loc, zero, index(type.getDimSize(axis)), one);
        b.setInsertionPointToStart(loop.getBody());
        offsets.push_back(loop.getInductionVar());
      }
      auto spatial =
          scf::ForOp::create(b, loc, zero, index(width), index(pixels));
      b.setInsertionPointToStart(spatial.getBody());
      Value x = spatial.getInductionVar();
      Value count = arith::MinSIOp::create(
          b, loc, index(pixels),
          arith::SubIOp::create(b, loc, index(width), x));
      // Preserve the source view's row stride, including cropped convolutions.
      auto transfer = [&](Value memory, Value bank, bool store) -> Value {
        int64_t elementBytes = store && quant ? 1 : 4;
        int64_t perRow = 16 / elementBytes;
        // A single pixel contains four adjacent bank rows. Transfer it as
        // one span instead of issuing one DMA command per quarter.
        if ((pixels == 1) && channels % 16 == 0) {
          SmallVector<OpFoldResult> address;
          for (Value offset : offsets)
            address.push_back(offset);
          address.push_back(x);
          address.push_back(channel);
          Value view = memref::SubViewOp::create(
              b, loc, memory, address,
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
          if (store)
            return BankMvoutStridedOp::create(
                b, loc, b.getI64Type(), view, bank, zero, index(elementBytes),
                index(16), 0, 1, index(16),
                arith::ConstantIndexOp::create(b, loc, 0));
          return BankMvinStridedOp::create(
              b, loc, b.getI64Type(), view, bank, zero, index(4), index(16), 0,
              1, index(16), arith::ConstantIndexOp::create(b, loc, 0));
        }
        auto metadata =
            memref::ExtractStridedMetadataOp::create(b, loc, memory);
        Value stride = pixels == 1
                           ? index(16)
                           : Value(arith::MulIOp::create(
                                 b, loc, metadata.getStrides()[rank - 2],
                                 index(elementBytes)));
        if (elementBytes == 4 && channels % 8 == 0) {
          Value lanes = arith::MinSIOp::create(
              b, loc, index(16),
              arith::SubIOp::create(b, loc, index(channels), channel));
          Value blockLog = arith::SelectOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lanes,
                                    index(16)),
              index(2), one);
          Value rows = arith::ShLIOp::create(b, loc, count, blockLog);
          SmallVector<OpFoldResult> address(offsets.begin(), offsets.end());
          address.push_back(x);
          address.push_back(channel);
          Value view = memref::SubViewOp::create(
              b, loc, memory, address,
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
          Value blockStride = pixels == 1 ? index(64) : stride;
          if (store)
            return BankMvoutStridedOp::create(b, loc, b.getI64Type(), view,
                                              bank, zero, rows, blockStride, 0,
                                              4, index(16), blockLog);
          return BankMvinStridedOp::create(b, loc, b.getI64Type(), view, bank,
                                           zero, rows, blockStride, 0, 4,
                                           index(16), blockLog);
        }
        Value quarters = arith::CeilDivUIOp::create(
            b, loc,
            arith::MinSIOp::create(
                b, loc, index(16),
                arith::SubIOp::create(b, loc, index(channels), channel)),
            index(perRow));
        auto loop = scf::ForOp::create(
            b, loc, zero, quarters, one, ValueRange{bank},
            [&](OpBuilder &builder, Location, Value quarter,
                ValueRange carried) {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPoint(builder.getInsertionBlock(),
                                  builder.getInsertionPoint());
              Value c = arith::AddIOp::create(
                  b, loc, channel,
                  arith::MulIOp::create(b, loc, quarter, index(perRow)));
              SmallVector<OpFoldResult> address;
              for (Value offset : offsets)
                address.push_back(offset);
              address.push_back(x);
              address.push_back(c);
              Value view = memref::SubViewOp::create(
                  b, loc, memory, address,
                  SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
                  SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
              Value bytes = arith::MulIOp::create(
                  b, loc, index(elementBytes),
                  arith::MinSIOp::create(
                      b, loc, index(perRow),
                      arith::SubIOp::create(b, loc, index(channels), c)));
              Value result;
              if (store)
                result = BankMvoutStridedOp::create(
                    b, loc, b.getI64Type(), view, carried[0], quarter, count,
                    stride, 0, elementBytes, bytes,
                    arith::ConstantIndexOp::create(b, loc, 0));
              else
                result = BankMvinStridedOp::create(
                    b, loc, b.getI64Type(), view, carried[0], quarter, count,
                    stride, 0, elementBytes, bytes,
                    arith::ConstantIndexOp::create(b, loc, 0));
              scf::YieldOp::create(b, loc, result);
            });
        bank = loop.getResult(0);
        return bank;
      };
      Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      a = BankClearOp::create(b, loc, b.getI64Type(), a, 1, 1);
      a = transfer(input, a, false);
      Value other = parameters;
      if (binary && !broadcast) {
        other = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        other = BankClearOp::create(b, loc, b.getI64Type(), other, 1, 1);
        other = transfer(rhs, other, false);
      }
      Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      if (floatingQuant)
        result = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
            b, loc, b.getI64Type(), a, parameters, result, pixels * 4);
      else if (quant)
        result = ::buddy::toint8_ball::BankInt32ToInt8Op::create(
            b, loc, b.getI64Type(), a, parameters, result, pixels, false);
      else
        result = ::buddy::fixed_ball::BankFixedOp::create(
            b, loc, b.getI64Type(), a, other, result, pixels * 4, mode,
            cast<IntegerAttr>(op->getAttr("fraction")).getInt(), broadcast,
            false);
      BankReleaseOp::create(b, loc, a);
      if (binary && !broadcast)
        BankReleaseOp::create(b, loc, other);
      result = transfer(output, result, true);
      BankReleaseOp::create(b, loc, result);
      b.setInsertionPoint(channelLoop.getBody()->getTerminator());
      BankReleaseOp::create(b, loc, parameters);
    }
    Value tensor =
        bufferization::ToTensorOp::create(b, loc, type, output, true, true);
    op->getResult(0).replaceAllUsesWith(tensor);
    op->erase();
  }
  return success();
}
