//===- LowerLinear.cpp - Matrix products in bank tiles --------------------===//
//
// C = bias + A @ W over sixteen output columns per matmul_i8. M goes in tiles
// of the matmul rows; K goes in calls of up to six groups of A and B, each
// group one reduction block. The host packs every operand in the bank layout
// of matmul_i8, so a whole-bank DMA fills all groups of a bank at once.
//
//===----------------------------------------------------------------------===//
#include "Buckyball/BuckyballOps.h"
#include "Custom/CustomOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "int2fpOps.h"
#include "mataddOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "smatmulOps.h"
#include "toint8Ops.h"

using namespace mlir;
using namespace ::buddy::buckyball;

pebble::LinearSplit pebble::linearSplit(int64_t k, int64_t rows,
                                        bool floating) {
  // The most groups of A (and of its FP32 source) and B in one matmul call;
  // the accumulator, a partial, their sum and the scales take four banks
  // more.
  constexpr int64_t callGroups = 6;
  int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
  // A group of A holds rows x k / 16 lines (four times that in FP32), B k.
  int64_t maxK = std::min(depth, depth * (floating ? 4 : 16) / rows) / 16 * 16;
  int64_t paddedK = (k + 15) / 16 * 16;
  int64_t blocks = (paddedK + maxK - 1) / maxK;
  int64_t calls = (blocks + callGroups - 1) / callGroups;
  int64_t groups = (blocks + calls - 1) / calls;
  int64_t blockK =
      ((paddedK + calls * groups - 1) / (calls * groups) + 15) / 16 * 16;
  return {calls, groups, blockK};
}

namespace {

// output (M x N) = bias + input (M x K) @ weight (K x N), converted with the
// per-column `scales` and `relu` to the element type of `output`: INT32 as
// is, FP32 by int32_to_fp32, INT8 by int32_to_int8. An FP32 input is
// quantized with `multiplier`. A resident input bank of one input row holds
// K in groups of `residentBlock` columns and replaces the input of a single
// call. `bias` has one row, or one row per row of the input. Without
// `output`, the INT8 row stays in the returned bank, one group per sixteen
// columns.
Value linearTiles(OpBuilder &builder, Location loc, Value input, Value resident,
                  int64_t residentBlock, Value multiplier, Value weight,
                  Value bias, Value scales, bool relu, Value output) {
  auto weightType = cast<MemRefType>(weight.getType());
  int64_t k = weightType.getDimSize(0), n = weightType.getDimSize(1);
  int64_t m = output ? cast<MemRefType>(output.getType()).getDimSize(0) : 1;
  bool floating = multiplier != nullptr;
  bool perRow = m > 1 && cast<MemRefType>(bias.getType()).getDimSize(0) == m;
  Type outElement = output ? cast<MemRefType>(output.getType()).getElementType()
                           : builder.getI8Type();
  int64_t rows = m == 1 ? 1 : pebble::tileRows();
  int64_t paddedK = (k + 15) / 16 * 16;
  pebble::LinearSplit split = pebble::linearSplit(k, rows, floating);
  int64_t calls = split.calls, groups = split.groups, blockK = split.blockK;
  if (resident) {
    calls = 1;
    blockK = residentBlock;
    groups = paddedK / residentBlock;
  }
  int64_t aLines = blockK / 16 * rows * (floating ? 4 : 1);
  int64_t aWidth = groups * (floating ? 4 : 16);
  int64_t slices = (n + 15) / 16;
  Type i64 = builder.getI64Type();
  auto index = [&](OpBuilder &b, int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  auto extent = [&](OpBuilder &b, int64_t size, int64_t limit,
                    Value start) -> Value {
    return arith::MinSIOp::create(
        b, loc, index(b, size),
        arith::SubIOp::create(b, loc, index(b, limit), start));
  };
  // Nested host loops over [0, ends) with the given body.
  auto loops = [&](OpBuilder &b, ArrayRef<Value> ends,
                   function_ref<void(OpBuilder &, ValueRange)> body) {
    OpBuilder::InsertionGuard guard(b);
    SmallVector<Value> indices;
    for (Value end : ends) {
      auto loop = scf::ForOp::create(b, loc, index(b, 0), end, index(b, 1));
      b.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
    }
    body(b, indices);
  };
  // The reduction block of column `kk`: call, group and column in the block.
  auto block = [&](OpBuilder &b, Value kk) {
    Value inBlock = arith::RemUIOp::create(b, loc, kk, index(b, blockK));
    Value blockIndex = arith::DivUIOp::create(b, loc, kk, index(b, blockK));
    Value call = arith::DivUIOp::create(b, loc, blockIndex, index(b, groups));
    Value group = arith::RemUIOp::create(b, loc, blockIndex, index(b, groups));
    return std::make_tuple(call, group, inBlock);
  };

  // B of every output slice and call, packed once: line l of group g holds
  // row g * blockK + l of the block.
  Value weightPack = memref::AllocOp::create(
      builder, loc,
      MemRefType::get({slices, calls, blockK, groups * 16},
                      builder.getI8Type()));
  linalg::FillOp::create(
      builder, loc,
      ValueRange{arith::ConstantIntOp::create(builder, loc, 0, 8)}, weightPack);
  loops(builder, {index(builder, k), index(builder, n)},
        [&](OpBuilder &b, ValueRange at) {
          auto [call, group, inBlock] = block(b, at[0]);
          Value value = memref::LoadOp::create(b, loc, weight, at);
          Value slice = arith::DivUIOp::create(b, loc, at[1], index(b, 16));
          Value lane = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, group, index(b, 16)),
              arith::RemUIOp::create(b, loc, at[1], index(b, 16)));
          memref::StoreOp::create(b, loc, value, weightPack,
                                  ValueRange{slice, call, inBlock, lane});
        });
  Value inputPack;
  if (!resident)
    inputPack = memref::AllocOp::create(
        builder, loc,
        MemRefType::get({calls, aLines, aWidth},
                        floating ? Type(builder.getF32Type())
                                 : builder.getI8Type()));
  Value biasPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value zeroPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value tilePack;
  if (perRow)
    tilePack = memref::AllocOp::create(
        builder, loc, MemRefType::get({rows, 16}, builder.getI32Type()));
  Value scalePack;
  if (scales)
    scalePack = memref::AllocOp::create(
        builder, loc, MemRefType::get({16}, builder.getF32Type()));
  Value outPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({rows, 16}, outElement));
  Value izero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  linalg::FillOp::create(builder, loc, izero, zeroPack);

  // A of reduction call `call`, quantized on the core from FP32.
  auto loadA = [&](OpBuilder &b, Value call) -> Value {
    Value slice = memref::SubViewOp::create(
        b, loc, inputPack,
        ArrayRef<OpFoldResult>{call, b.getIndexAttr(0), b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(aLines),
                               b.getIndexAttr(aWidth)},
        SmallVector<OpFoldResult>(3, b.getIndexAttr(1)));
    Value a = BankAllocOp::create(b, loc, i64, 1, groups);
    if (!floating)
      return BankMvinOp::create(b, loc, i64, slice, a, aLines, 1);
    Value values = BankAllocOp::create(b, loc, i64, 1, groups);
    values = BankMvinOp::create(b, loc, i64, slice, values, aLines, 1);
    a = ::buddy::toint8_ball::BankFp32ToInt8Op::create(b, loc, i64, values, a,
                                                       multiplier, aLines);
    BankReleaseOp::create(b, loc, values);
    return a;
  };
  // Reduction call `call` of output slice `slice` into `destination`, read
  // as its bias. `a` is the A of the call, or null to load it.
  auto product = [&](OpBuilder &b, Value slice, Value call, Value a,
                     Value destination) -> Value {
    Value loaded = a ? a : loadA(b, call);
    Value w = BankAllocOp::create(b, loc, i64, 1, groups);
    w = BankMvinOp::create(
        b, loc, i64,
        memref::SubViewOp::create(
            b, loc, weightPack,
            ArrayRef<OpFoldResult>{slice, call, b.getIndexAttr(0),
                                   b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                   b.getIndexAttr(blockK),
                                   b.getIndexAttr(groups * 16)},
            SmallVector<OpFoldResult>(4, b.getIndexAttr(1))),
        w, blockK, 1);
    Value result = ::buddy::smatmul_ball::BankMatmulI8Op::create(
        b, loc, i64, loaded, w, destination, rows, blockK);
    if (!a)
      BankReleaseOp::create(b, loc, loaded);
    BankReleaseOp::create(b, loc, w);
    return result;
  };
  // sum = acc + partial; add_i32 writes a third bank.
  auto add = [&](OpBuilder &b, Value acc, Value partial) -> Value {
    Value sum = BankAllocOp::create(b, loc, i64, 1, 1);
    sum = ::buddy::matadd_ball::BankAddI32Op::create(b, loc, i64, acc, partial,
                                                     sum, rows * 4);
    BankReleaseOp::create(b, loc, acc);
    BankReleaseOp::create(b, loc, partial);
    return sum;
  };

  Value vector;
  pebble::repeat(
      builder, loc, 0, m, rows, false, {},
      [&](OpBuilder &b, Value row, ValueRange) -> SmallVector<Value> {
        Value valid = extent(b, rows, m, row);
        if (!resident) {
          linalg::FillOp::create(
              b, loc,
              ValueRange{
                  floating
                      ? arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0))
                            .getResult()
                      : arith::ConstantIntOp::create(b, loc, 0, 8).getResult()},
              inputPack);
          loops(b, {valid, index(b, k)}, [&](OpBuilder &b, ValueRange at) {
            auto [call, group, inBlock] = block(b, at[1]);
            Value value = memref::LoadOp::create(
                b, loc, input,
                ValueRange{arith::AddIOp::create(b, loc, at[0], row), at[1]});
            // Slice t = inBlock / 16 of M row r is line t * rows + r.
            Value line = arith::AddIOp::create(
                b, loc,
                arith::MulIOp::create(
                    b, loc,
                    arith::DivUIOp::create(b, loc, inBlock, index(b, 16)),
                    index(b, rows)),
                at[0]);
            Value lane = arith::RemUIOp::create(b, loc, inBlock, index(b, 16));
            if (floating) {
              line = arith::AddIOp::create(
                  b, loc, arith::MulIOp::create(b, loc, line, index(b, 4)),
                  arith::DivUIOp::create(b, loc, lane, index(b, 4)));
              lane = arith::RemUIOp::create(b, loc, lane, index(b, 4));
            }
            lane = arith::AddIOp::create(
                b, loc,
                arith::MulIOp::create(b, loc, group,
                                      index(b, floating ? 4 : 16)),
                lane);
            memref::StoreOp::create(b, loc, value, inputPack,
                                    ValueRange{call, line, lane});
          });
        }
        // One call keeps A in its bank for every output slice.
        Value a = resident;
        if (!resident && calls == 1)
          a = loadA(b, index(b, 0));
        auto outputSlice = [&](OpBuilder &b,
                               Value column) -> SmallVector<Value> {
          Value columns = extent(b, 16, n, column);
          Value slice = arith::DivUIOp::create(b, loc, column, index(b, 16));
          linalg::FillOp::create(b, loc, izero, biasPack);
          if (!perRow)
            loops(b, {columns}, [&](OpBuilder &b, ValueRange at) {
              Value value = memref::LoadOp::create(
                  b, loc, bias,
                  ValueRange{index(b, 0),
                             arith::AddIOp::create(b, loc, at[0], column)});
              memref::StoreOp::create(b, loc, value, biasPack, at[0]);
            });
          Value scaleBank;
          if (scales) {
            linalg::FillOp::create(b, loc,
                                   ValueRange{arith::ConstantOp::create(
                                       b, loc, b.getF32FloatAttr(1))},
                                   scalePack);
            loops(b, {columns}, [&](OpBuilder &b, ValueRange at) {
              Value value = memref::LoadOp::create(
                  b, loc, scales,
                  ValueRange{index(b, 0),
                             arith::AddIOp::create(b, loc, at[0], column)});
              memref::StoreOp::create(b, loc, value, scalePack, at[0]);
            });
            scaleBank = BankAllocOp::create(b, loc, i64, 1, 1);
            scaleBank =
                BankMvinOp::create(b, loc, i64, scalePack, scaleBank, 4, 1);
          }
          Value acc = BankAllocOp::create(b, loc, i64, 1, 1);
          acc = BankMvinOp::create(b, loc, i64, biasPack, acc, 4, 1);
          acc = product(b, slice, index(b, 0), a, acc);
          if (rows == 1) {
            // One row accumulates in place: the old C is the bias.
            acc =
                pebble::repeat(b, loc, 1, calls, 1, false, {acc},
                               [&](OpBuilder &b, Value call, ValueRange args) {
                                 return SmallVector<Value>{
                                     product(b, slice, call, Value(), args[0])};
                               })[0];
          } else {
            for (int64_t call = 1; call < calls; ++call) {
              Value partial = BankAllocOp::create(b, loc, i64, 1, 1);
              partial =
                  BankMvinOp::create(b, loc, i64, zeroPack, partial, 4, 1);
              acc = add(b, acc,
                        product(b, slice, index(b, call), Value(), partial));
            }
          }
          if (perRow) {
            linalg::FillOp::create(b, loc, izero, tilePack);
            loops(b, {valid, columns}, [&](OpBuilder &b, ValueRange at) {
              Value value = memref::LoadOp::create(
                  b, loc, bias,
                  ValueRange{arith::AddIOp::create(b, loc, at[0], row),
                             arith::AddIOp::create(b, loc, at[1], column)});
              memref::StoreOp::create(b, loc, value, tilePack, at);
            });
            Value initial = BankAllocOp::create(b, loc, i64, 1, 1);
            initial =
                BankMvinOp::create(b, loc, i64, tilePack, initial, rows * 4, 1);
            acc = add(b, acc, initial);
          }
          Value restored = acc;
          if (scales) {
            restored = BankAllocOp::create(b, loc, i64, 1, 1);
            restored =
                outElement.isF32()
                    ? Value(::buddy::int2fp_ball::BankInt32ToFp32Op::create(
                          b, loc, i64, acc, scaleBank, restored, rows, relu))
                    : Value(::buddy::toint8_ball::BankInt32ToInt8Op::create(
                          b, loc, i64, acc, scaleBank, restored, rows, relu));
            BankReleaseOp::create(b, loc, acc);
            BankReleaseOp::create(b, loc, scaleBank);
          }
          if (!output)
            return {restored};
          restored =
              BankMvoutOp::create(b, loc, i64, outPack, restored,
                                  outElement.isInteger(8) ? rows : rows * 4, 1);
          BankReleaseOp::create(b, loc, restored);
          loops(b, {valid, columns}, [&](OpBuilder &b, ValueRange at) {
            Value value = memref::LoadOp::create(b, loc, outPack, at);
            memref::StoreOp::create(
                b, loc, value, output,
                ValueRange{arith::AddIOp::create(b, loc, at[0], row),
                           arith::AddIOp::create(b, loc, at[1], column)});
          });
          return {};
        };
        if (output) {
          pebble::repeat(b, loc, 0, n, 16, false, {},
                         [&](OpBuilder &b, Value column, ValueRange) {
                           return outputSlice(b, column);
                         });
        } else {
          // Each output slice appends its group to the resident row.
          for (int64_t column = 0; column < n; column += 16) {
            Value slice = outputSlice(b, index(b, column))[0];
            vector = vector ? BankTransferOp::create(b, loc, i64, slice, vector)
                                  .getResult()
                            : slice;
          }
        }
        if (a)
          BankReleaseOp::create(b, loc, a);
        return {};
      });
  return vector;
}
} // namespace

LogicalResult pebble::lowerLinear(::buddy::custom::CustomOp op,
                                  Value residentInput, Value &result,
                                  Value &resident) {
  StringRef kind = op.getKind();
  bool integer = kind == "matmul_i8", requant = kind == "linear_i8_requant";
  // A fused linear_i8 requantizes with the multiplier of its consumer.
  ValueRange args = op.getInputs();
  bool fused = kind == "linear_i8" && args.size() == 7;
  if ((!fused && args.size() != (integer   ? 3
                                 : requant ? 4
                                           : 6)) ||
      op.getNumResults() != 1)
    return op.emitOpError("has the wrong number of inputs or results for ")
           << kind;
  auto a = cast<RankedTensorType>(args[0].getType());
  auto w = cast<RankedTensorType>(args[1].getType());
  auto biasType = cast<RankedTensorType>(args[2].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  int64_t m = resultType.getDimSize(0), n = w.getDimSize(1);
  bool codes = integer || requant || fused;
  if (!a.hasStaticShape() || !w.hasStaticShape() || a.getRank() != 2 ||
      w.getRank() != 2 || a.getDimSize(1) != w.getDimSize(0) ||
      a.getDimSize(0) != m || !w.getElementType().isInteger(8) ||
      !biasType.getElementType().isInteger(32) ||
      biasType.getShape() != ArrayRef<int64_t>({integer ? m : 1, n}) ||
      resultType.getShape() != ArrayRef<int64_t>({m, n}) ||
      !(integer ? resultType.getElementType().isInteger(32)
        : codes ? resultType.getElementType().isInteger(8)
                : resultType.getElementType().isF32()) ||
      ((integer || requant) && !a.getElementType().isInteger(8)))
    return op.emitOpError("requires matching MxK/KxN tensors and INT32 bias "
                          "for ")
           << kind;
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value input, multiplier, scales;
  if (!residentInput)
    input = tensorBuffer(b, loc, args[0]);
  if (!integer && !requant && a.getElementType().isF32())
    multiplier =
        tensor::ExtractOp::create(b, loc, args[3], ValueRange{zero, zero});
  if (requant) {
    scales = tensorBuffer(b, loc, args[3]);
  } else if (!integer) {
    // input_step * weight_step per column restores FP32 values; a fused
    // Linear scales them to the codes of its consumer.
    Value step =
        tensor::ExtractOp::create(b, loc, args[4], ValueRange{zero, zero});
    if (fused)
      step = arith::MulFOp::create(
          b, loc, step,
          tensor::ExtractOp::create(b, loc, args[6], ValueRange{zero, zero}));
    Value weightSteps = tensorBuffer(b, loc, args[5]);
    scales = memref::AllocOp::create(b, loc,
                                     MemRefType::get({1, n}, b.getF32Type()));
    auto loop = scf::ForOp::create(b, loc, zero,
                                   arith::ConstantIndexOp::create(b, loc, n),
                                   arith::ConstantIndexOp::create(b, loc, 1));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value c = loop.getInductionVar();
    Value product = arith::MulFOp::create(
        b, loc, step,
        memref::LoadOp::create(b, loc, weightSteps, ValueRange{zero, c}));
    memref::StoreOp::create(b, loc, product, scales, ValueRange{zero, c});
  }
  // A flattened map is one block of K; a resident Linear row has a group per
  // sixteen columns.
  auto producer = args[0].getDefiningOp<::buddy::custom::CustomOp>();
  int64_t residentBlock =
      producer && producer.getKind() == "flatten_nchw" ? a.getDimSize(1) : 16;
  bool relu =
      (requant || fused) && cast<BoolAttr>(op->getAttr("relu")).getValue();
  Value output;
  if (!op->hasAttr("custom.resident"))
    output = memref::AllocOp::create(
        b, loc, MemRefType::get({m, n}, resultType.getElementType()));
  resident = linearTiles(b, loc, input, residentInput, residentBlock,
                         multiplier, tensorBuffer(b, loc, args[1]),
                         tensorBuffer(b, loc, args[2]), scales, relu, output);
  if (output)
    result = bufferization::ToTensorOp::create(b, loc, resultType, output, true,
                                               true);
  return success();
}

// The mean of every channel: ones (1 x pixels) @ input (pixels x channels),
// requantized with input_scale / (pixels * output_scale).
LogicalResult pebble::lowerGlobalAverage(::buddy::custom::CustomOp op,
                                         Value &result) {
  auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (!type.hasStaticShape() || type.getRank() != 4 ||
      type.getDimSize(0) != 1 || !type.getElementType().isInteger(8) ||
      !resultType.getElementType().isInteger(8) ||
      resultType.getShape() != ArrayRef<int64_t>({1, 1, 1, type.getDimSize(3)}))
    return op.emitOpError(
        "global_avg_i8 requires NHWC INT8 input and 1x1x1xC INT8 output");
  int64_t pixels = type.getDimSize(1) * type.getDimSize(2),
          channels = type.getDimSize(3);
  float ratio =
      cast<FloatAttr>(op->getAttr("input_scale")).getValueAsDouble() /
      (pixels *
       cast<FloatAttr>(op->getAttr("output_scale")).getValueAsDouble());
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value ones = memref::AllocOp::create(
      b, loc, MemRefType::get({1, pixels}, b.getI8Type()));
  linalg::FillOp::create(
      b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 1, 8)}, ones);
  Value bias = memref::AllocOp::create(
      b, loc, MemRefType::get({1, channels}, b.getI32Type()));
  linalg::FillOp::create(
      b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 0, 32)}, bias);
  Value scales = memref::AllocOp::create(
      b, loc, MemRefType::get({1, channels}, b.getF32Type()));
  linalg::FillOp::create(
      b, loc,
      ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(ratio))},
      scales);
  Value weight = memref::CollapseShapeOp::create(
      b, loc, tensorBuffer(b, loc, op.getInputs()[0]),
      ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}});
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get({1, channels}, b.getI8Type()));
  linearTiles(b, loc, ones, Value(), 0, Value(), weight, bias, scales, false,
              output);
  Value flat = bufferization::ToTensorOp::create(
      b, loc, RankedTensorType::get({1, channels}, b.getI8Type()), output, true,
      true);
  result = tensor::ExpandShapeOp::create(
      b, loc, resultType, flat, ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}});
  return success();
}
