//===- LowerLinear.cpp - Matrix products in bank tiles --------------------===//
//
// C = bias + A @ W over sixteen output columns per matmul_i8. M goes in tiles
// of the matmul rows; K goes in calls of up to six groups of A and B, each
// group one reduction block. Every group of B is a strided DMA of the [K, N]
// weight in place. INT8 rows move directly into A with strided DMA;
// FP32 input is packed before conversion.
//
//===----------------------------------------------------------------------===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "int2fpOps.h"
#include "mataddOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "smatmulOps.h"
#include "toint8Ops.h"

using namespace mlir;
using namespace ::buddy::acc;

pebble::LinearSplit pebble::linearSplit(int64_t k, int64_t rows,
                                        bool floating) {
  // The most groups of A (and of its FP32 source) and B in one matmul call;
  // the accumulator, a partial, their sum and the scales take four banks
  // more.
  constexpr int64_t callGroups = 6;
  int64_t depth = target::getTarget().bankDepth;
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

// output (M x N) = bias + input (M x K) @ weight (K x N), converted with the
// per-column `scales` and `relu` to the element type of `output`: INT32 as
// is, FP32 by int32_to_fp32, INT8 by int32_to_int8. An FP32 input is
// quantized with `multiplier`, one for the whole input: a group of A holds
// several sixteen-column slices of K, and fp32_to_int8 scales every row of a
// group with the same sixteen multipliers. A resident input bank of one input
// row holds K in groups of `residentBlock` columns and replaces the input of a
// single call. `weight` is read in place: N is a multiple of sixteen, and
// the rows of a group past K stay unwritten, multiplied by the zero columns
// of A. `bias` has one row, or one row per row of the input. Without
// `output`, the INT8 row stays in the returned bank, one group per sixteen
// columns.
Value pebble::linearTiles(OpBuilder &builder, Location loc, Value input,
                          Value resident, int64_t residentBlock,
                          Value multiplier, Value weight, Value bias,
                          Value scales, bool relu, Value output) {
  auto weightType = cast<MemRefType>(weight.getType());
  bool packedWeight = weightType.getRank() == 3;
  int64_t k = weightType.getDimSize(packedWeight ? 1 : 0);
  int64_t n =
      packedWeight ? weightType.getDimSize(0) * 16 : weightType.getDimSize(1);
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
  bool directInput = !floating;
  int64_t aLines = blockK / 16 * rows * (floating ? 4 : 1);
  int64_t aWidth = groups * (floating ? 4 : 16);
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

  Value inputPack;
  if (!resident && !directInput)
    inputPack = memref::AllocOp::create(
        builder, loc,
        MemRefType::get({calls, aLines, aWidth},
                        floating ? Type(builder.getF32Type())
                                 : builder.getI8Type()));
  Value zeroPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value tilePack;
  if (perRow)
    tilePack = memref::AllocOp::create(
        builder, loc, MemRefType::get({rows, 16}, builder.getI32Type()));
  // The fp32_to_int8 multipliers: rows 4g..4g+3 of one group for group g of
  // A.
  Value multiplierPack;
  if (floating) {
    multiplierPack = memref::AllocOp::create(
        builder, loc, MemRefType::get({groups * 4, 4}, builder.getF32Type()));
    linalg::FillOp::create(builder, loc, multiplier, multiplierPack);
  }
  Value izero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  linalg::FillOp::create(builder, loc, izero, zeroPack);

  // A of reduction call `call`, quantized on the hart from FP32.
  auto loadA = [&](OpBuilder &b, int64_t call, Value firstRow) -> Value {
    if (directInput) {
      Value a = BankAllocOp::create(b, loc, i64, 1, groups);
      if (m == 1) {
        if (k != calls * groups * blockK)
          a = BankClearOp::create(b, loc, i64, a, 1, groups);
        for (int64_t group = 0; group < groups; ++group) {
          int64_t first = (call * groups + group) * blockK;
          int64_t count = std::min(blockK, k - first);
          if (count <= 0)
            break;
          if (count / 16) {
            Value memory = memref::SubViewOp::create(
                b, loc, input,
                ArrayRef<OpFoldResult>{firstRow, b.getIndexAttr(first)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                       b.getIndexAttr(count / 16 * 16)},
                SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
            a = BankMvinGroupOp::create(b, loc, i64, memory, a, group,
                                        count / 16, 1);
          }
          if (count % 16) {
            Value memory = memref::SubViewOp::create(
                b, loc, input,
                ArrayRef<OpFoldResult>{firstRow,
                                       b.getIndexAttr(first + count / 16 * 16)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                       b.getIndexAttr(count % 16)},
                SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
            a = BankMvinStridedOp::create(
                b, loc, i64, memory, a, index(b, count / 16), index(b, 1),
                index(b, 16), group, 1, index(b, count % 16), index(b, 0));
          }
        }
        return a;
      }
      if (m % rows || k != calls * groups * blockK)
        a = BankClearOp::create(b, loc, i64, a, 1, groups);
      Value valid = extent(b, rows, m, firstRow);
      int64_t dmaRows =
          k > 4095
              ? 1
              : 1 + ((target::getTarget().dmaMaxRanges - 1) * 4096 - 16) / k;
      for (int64_t group = 0; group < groups; ++group) {
        int64_t base = (call * groups + group) * blockK;
        for (int64_t kk = 0; kk < blockK && base + kk < k; kk += 16) {
          auto loop = scf::ForOp::create(b, loc, index(b, 0), valid,
                                         index(b, dmaRows), ValueRange{a});
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(loop.getBody());
            Value at = loop.getInductionVar();
            Value sourceRow = arith::AddIOp::create(b, loc, firstRow, at);
            Value memory = memref::SubViewOp::create(
                b, loc, input,
                ArrayRef<OpFoldResult>{sourceRow, b.getIndexAttr(base + kk)},
                SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
                SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
            Value bankRow =
                arith::AddIOp::create(b, loc, index(b, kk / 16 * rows), at);
            Value count = arith::MinSIOp::create(
                b, loc, index(b, dmaRows),
                arith::SubIOp::create(b, loc, valid, at));
            Value loaded = BankMvinStridedOp::create(
                b, loc, i64, memory, loop.getRegionIterArgs()[0], bankRow,
                count, index(b, k > 4095 ? 16 : k), group, 1,
                index(b, std::min(int64_t(16), k - base - kk)), index(b, 0));
            scf::YieldOp::create(b, loc, loaded);
          }
          a = loop.getResult(0);
        }
      }
      return a;
    }
    Value slice = memref::SubViewOp::create(
        b, loc, inputPack,
        ArrayRef<OpFoldResult>{index(b, call), b.getIndexAttr(0),
                               b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(aLines),
                               b.getIndexAttr(aWidth)},
        SmallVector<OpFoldResult>(3, b.getIndexAttr(1)));
    Value a = BankAllocOp::create(b, loc, i64, 1, groups);
    if (!floating)
      return BankMvinOp::create(b, loc, i64, slice, a, aLines, 1);
    Value values = BankAllocOp::create(b, loc, i64, 1, groups);
    values = BankMvinOp::create(b, loc, i64, slice, values, aLines, 1);
    Value multipliers = BankAllocOp::create(b, loc, i64, 1, 1);
    multipliers = BankMvinOp::create(b, loc, i64, multiplierPack, multipliers,
                                     groups * 4, 1);
    a = ::buddy::toint8_ball::BankFp32ToInt8Op::create(b, loc, i64, values,
                                                       multipliers, a, aLines);
    BankReleaseOp::create(b, loc, values);
    BankReleaseOp::create(b, loc, multipliers);
    return a;
  };
  // Reduction call `call` of the output slice at `column` into
  // `destination`, read as its bias. `a` is the A of the call, or null to
  // load it. Line l of group g of B is row (call * groups + g) * blockK + l
  // of the weight.
  auto product = [&](OpBuilder &b, Value row, Value column, int64_t call,
                     Value a, Value destination) -> Value {
    Value loaded = a ? a : loadA(b, call, row);
    Value w = BankAllocOp::create(b, loc, i64, 1, groups);
    for (int64_t group = 0; group < groups; ++group) {
      int64_t first = (call * groups + group) * blockK;
      int64_t depth = std::min(blockK, k - first);
      if (depth <= 0)
        break;
      int64_t dmaRows =
          n > 4095
              ? 1
              : 1 + ((target::getTarget().dmaMaxRanges - 1) * 4096 - 16) / n;
      if (!packedWeight && depth > dmaRows) {
        for (int64_t start = 0; start < depth; start += dmaRows) {
          Value memory = memref::SubViewOp::create(
              b, loc, weight,
              ArrayRef<OpFoldResult>{b.getIndexAttr(first + start), column},
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
          w = BankMvinStridedOp::create(
              b, loc, i64, memory, w, index(b, start),
              index(b, std::min(dmaRows, depth - start)),
              index(b, n > 4095 ? 16 : n), group, 1, index(b, 16), index(b, 0));
        }
        continue;
      }
      Value memory;
      if (packedWeight) {
        memory = memref::SubViewOp::create(
            b, loc, weight,
            ArrayRef<OpFoldResult>{
                arith::DivUIOp::create(b, loc, column, index(b, 16))
                    .getResult(),
                b.getIndexAttr(first), b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(depth),
                                   b.getIndexAttr(16)},
            SmallVector<OpFoldResult>(3, b.getIndexAttr(1)));
      } else {
        memory = memref::SubViewOp::create(
            b, loc, weight,
            ArrayRef<OpFoldResult>{b.getIndexAttr(first), column},
            ArrayRef<OpFoldResult>{b.getIndexAttr(depth), b.getIndexAttr(16)},
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
      }
      w = BankMvinGroupOp::create(b, loc, i64, memory, w, group, depth,
                                  packedWeight ? 1 : n / 16);
    }
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
        if (!resident && !directInput) {
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
          a = loadA(b, 0, row);
        auto outputSlice = [&](OpBuilder &b,
                               Value column) -> SmallVector<Value> {
          Value columns = extent(b, 16, n, column);
          Value scaleBank;
          if (scales) {
            Value slice = memref::SubViewOp::create(
                b, loc, scales, ArrayRef<OpFoldResult>{index(b, 0), column},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(16)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
            scaleBank = BankAllocOp::create(b, loc, i64, 1, 1);
            scaleBank = BankMvinOp::create(b, loc, i64, slice, scaleBank, 4, 1);
          }
          Value acc = BankAllocOp::create(b, loc, i64, 1, 1);
          Value initial = zeroPack;
          if (!perRow)
            initial = memref::SubViewOp::create(
                b, loc, bias, ArrayRef<OpFoldResult>{index(b, 0), column},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(16)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          acc = BankMvinOp::create(b, loc, i64, initial, acc, 4, 1);
          acc = product(b, row, column, 0, a, acc);
          // One row accumulates in place: the old C is the bias.
          for (int64_t call = 1; call < calls; ++call) {
            if (rows == 1) {
              acc = product(b, row, column, call, Value(), acc);
              continue;
            }
            Value partial = BankAllocOp::create(b, loc, i64, 1, 1);
            partial = BankMvinOp::create(b, loc, i64, zeroPack, partial, 4, 1);
            acc = add(b, acc, product(b, row, column, call, Value(), partial));
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
          int64_t lines = outElement.isInteger(8) ? 1 : 4;
          if (m == 1) {
            loops(b, {valid}, [&](OpBuilder &b, ValueRange at) {
              Value memory = memref::SubViewOp::create(
                  b, loc, output,
                  ArrayRef<OpFoldResult>{
                      arith::AddIOp::create(b, loc, row, at[0]).getResult(),
                      column},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(16)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
              BankMvoutStridedOp::create(
                  b, loc, i64, memory, restored,
                  arith::MulIOp::create(b, loc, at[0], index(b, lines)),
                  index(b, lines), index(b, 16), 0, 1, index(b, 16),
                  arith::ConstantIndexOp::create(b, loc, 0));
            });
          } else {
            int64_t elementBytes = outElement.getIntOrFloatBitWidth() / 8;
            int64_t pitch = n * elementBytes;
            int64_t dmaRows =
                pitch > 4095
                    ? 1
                    : 1 + ((target::getTarget().dmaMaxRanges - 1) * 4096 -
                           16 * elementBytes) /
                              pitch;
            auto loop =
                scf::ForOp::create(b, loc, index(b, 0), valid,
                                   index(b, dmaRows), ValueRange{restored});
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(loop.getBody());
              Value at = loop.getInductionVar();
              Value memory = memref::SubViewOp::create(
                  b, loc, output,
                  ArrayRef<OpFoldResult>{
                      arith::AddIOp::create(b, loc, row, at).getResult(),
                      column},
                  SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
                  SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
              Value count = arith::MinSIOp::create(
                  b, loc, index(b, dmaRows),
                  arith::SubIOp::create(b, loc, valid, at));
              Value stored = BankMvoutStridedOp::create(
                  b, loc, i64, memory, loop.getRegionIterArgs()[0],
                  arith::MulIOp::create(b, loc, at, index(b, lines)),
                  arith::MulIOp::create(b, loc, count, index(b, lines)),
                  index(b, pitch > 4095 ? 16 * elementBytes : pitch), 0, lines,
                  index(b, 16), index(b, lines == 4 ? 2 : 0));
              scf::YieldOp::create(b, loc, stored);
            }
            restored = loop.getResult(0);
          }
          BankReleaseOp::create(b, loc, restored);
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
LogicalResult pebble::lowerLinear(::buddy::acc::CustomOp op,
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
  if (n % 16)
    return op.emitOpError("requires N a multiple of sixteen, the weight is "
                          "read in place");
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
  if (!integer && !requant && a.getElementType().isF32()) {
    if (cast<RankedTensorType>(args[3].getType()).getNumElements() != 1)
      return op.emitOpError("requires one input multiplier for the whole "
                            "input");
    multiplier =
        tensor::ExtractOp::create(b, loc, args[3], ValueRange{zero, zero});
  }
  if (requant) {
    scales = tensorBuffer(b, loc, args[3]);
  } else if (!integer) {
    if (fused &&
        cast<RankedTensorType>(args[6].getType()).getNumElements() != 1)
      return op.emitOpError(
          "requires one output multiplier for the whole result");
    DenseFPElementsAttr inputStep, weightStep, outputMultiplier;
    if (matchPattern(args[4], m_Constant(&inputStep)) &&
        matchPattern(args[5], m_Constant(&weightStep)) &&
        (!fused || matchPattern(args[6], m_Constant(&outputMultiplier)))) {
      APFloat step = *inputStep.getValues<APFloat>().begin();
      if (fused)
        step.multiply(*outputMultiplier.getValues<APFloat>().begin(),
                      APFloat::rmNearestTiesToEven);
      SmallVector<APFloat> products;
      for (APFloat weight : weightStep.getValues<APFloat>()) {
        APFloat product = step;
        product.multiply(weight, APFloat::rmNearestTiesToEven);
        products.push_back(product);
      }
      Value constant = arith::ConstantOp::create(
          b, loc,
          DenseFPElementsAttr::get(
              RankedTensorType::get({1, n}, b.getF32Type()), products));
      scales = tensorBuffer(b, loc, constant);
    } else {
      // input_step * weight_step per column restores FP32 values; a fused
      // Linear scales them to the codes of its consumer.
      Value step =
          tensor::ExtractOp::create(b, loc, args[4], ValueRange{zero, zero});
      if (fused) {
        step = arith::MulFOp::create(
            b, loc, step,
            tensor::ExtractOp::create(b, loc, args[6], ValueRange{zero, zero}));
      }
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
  }
  // A flattened map is one block of K; a resident Linear row has a group per
  // sixteen columns.
  auto producer = args[0].getDefiningOp<::buddy::acc::CustomOp>();
  int64_t residentBlock =
      producer && producer.getKind() == "flatten_nchw" ? a.getDimSize(1) : 16;
  bool relu =
      (requant || fused) && cast<BoolAttr>(op->getAttr("relu")).getValue();
  Value output;
  if (!op->hasAttr("bbacc.resident"))
    output = memref::AllocOp::create(
        b, loc, MemRefType::get({m, n}, resultType.getElementType()));
  Value weight = args[1];
  // The recipe's [N/16, K, 16] parameter viewed as the logical [K, N]
  // matrix. Read its contiguous column blocks without materializing the view.
  if (auto collapse = weight.getDefiningOp<tensor::CollapseShapeOp>()) {
    auto transpose = collapse.getSrc().getDefiningOp<linalg::TransposeOp>();
    if (transpose && transpose.getPermutation() == ArrayRef<int64_t>{1, 0, 2} &&
        collapse.getReassociationIndices() ==
            SmallVector<ReassociationIndices>{{0}, {1, 2}} &&
        cast<RankedTensorType>(transpose.getInput().getType()).getShape() ==
            ArrayRef<int64_t>{n / 16, a.getDimSize(1), 16})
      weight = transpose.getInput();
  }
  resident = linearTiles(b, loc, input, residentInput, residentBlock,
                         multiplier, tensorBuffer(b, loc, weight),
                         tensorBuffer(b, loc, args[2]), scales, relu, output);
  if (output)
    result = bufferization::ToTensorOp::create(b, loc, resultType, output, true,
                                               true);
  return success();
}

// The mean of every channel: ones (1 x pixels) @ input (pixels x channels),
// requantized with input_scale / (pixels * output_scale).
LogicalResult pebble::lowerGlobalAverage(::buddy::acc::CustomOp op,
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
  if (channels % 16)
    return op.emitOpError("global_avg_i8 requires a multiple of sixteen "
                          "channels");
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
