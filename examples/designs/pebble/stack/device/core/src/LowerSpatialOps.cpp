//===- LowerSpatialOps.cpp - Convolutions in bank tiles -------------------===//
//
// A convolution computes sixteen output channels per matmul_i8 over a tile of
// output pixels. The input tile, sixteen channels per group, goes through
// im2col_i8 into the A operand; the host packs the weights in the same K
// order into B. Input channels beyond one call form chunks whose partial
// products add up with add_i32. A fused convolution requantizes for its
// consumer and pools; when the consumer reads the whole result, the result
// stays in banks, one group per channel block.
//
//===----------------------------------------------------------------------===//
#include "Buckyball/BuckyballOps.h"
#include "Custom/CustomOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "im2colOps.h"
#include "int2fpOps.h"
#include "mataddOps.h"
#include "maxpoolOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "smatmulOps.h"
#include "toint8Ops.h"
#include "transposeOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using Computation = ::buddy::custom::CustomOp;

namespace {
// Lowers `op`. A resident input is a bank with one group per input channel
// block holding the whole input map. With custom.resident the INT8 result is
// returned in `image`, one group per output channel block; otherwise in
// `tensorOutput`.
LogicalResult convolution(Computation op, Value residentInput, Value &image,
                          Value &tensorOutput) {
  bool depthwise = op.getKind() == "depthwise_conv2d_i8";
  bool fused = op.getInputs().size() == 7;
  bool resident = op->hasAttr("custom.resident");
  auto args = op.getInputs();
  auto inputType = cast<RankedTensorType>(args[0].getType());
  auto weightType = cast<RankedTensorType>(args[1].getType());
  auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
  auto strides = cast<DenseI64ArrayAttr>(op->getAttr("strides")).asArrayRef();
  auto padding = cast<DenseI64ArrayAttr>(op->getAttr("padding")).asArrayRef();
  if (args.size() < 6 || !inputType.hasStaticShape() ||
      inputType.getRank() != 4 || !weightType.hasStaticShape() ||
      weightType.getRank() != 4 || inputType.getDimSize(0) != 1 ||
      !weightType.getElementType().isInteger(8) ||
      !cast<RankedTensorType>(args[2].getType()).getElementType().isInteger(32))
    return op.emitOpError("requires static NHWC/FCHW convolution with INT32 "
                          "bias");
  bool floating = inputType.getElementType().isF32();
  int64_t pool = fused ? cast<IntegerAttr>(op->getAttr("pool")).getInt() : 1;
  bool relu = fused && cast<BoolAttr>(op->getAttr("relu")).getValue();
  int64_t cin = inputType.getDimSize(3), cout = weightType.getDimSize(0);
  int64_t kernel = weightType.getDimSize(2), stride = strides[0];
  int64_t height = outputType.getDimSize(1) * pool,
          width = outputType.getDimSize(2) * pool;
  int64_t blocksIn = (cin + 15) / 16;
  // The tile, the input tile around it and the channel blocks per chunk. A
  // resident input is one tile of the whole map that im2col pads itself.
  int64_t side, inputSide, chunk, im2colPadding;
  if (residentInput) {
    side = height;
    inputSide = inputType.getDimSize(1);
    chunk = blocksIn;
    im2colPadding = padding[0];
  } else {
    pebble::ConvTiling tiling =
        pebble::convTiling(height / pool, width / pool, pool, stride, kernel,
                           blocksIn, depthwise, floating);
    side = tiling.side;
    inputSide = tiling.inputSide;
    chunk = tiling.chunk;
    im2colPadding = 0;
  }
  int64_t rows = (side * side + pebble::tileRows() - 1) / pebble::tileRows() *
                 pebble::tileRows();
  int64_t pooled = side / pool;
  // Chunks of input channel blocks: [start, start + blocks). Depthwise reads
  // the block of its output channels.
  SmallVector<std::pair<int64_t, int64_t>> chunks;
  for (int64_t start = 0; start < (depthwise ? 1 : blocksIn); start += chunk)
    chunks.push_back({start, std::min(chunk, blocksIn - start)});

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Type i64 = builder.getI64Type();
  auto index = [&](OpBuilder &b, int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  auto extent = [&](OpBuilder &b, int64_t size, int64_t limit,
                    Value start) -> Value {
    return arith::MinSIOp::create(
        b, loc, index(b, size),
        arith::SubIOp::create(b, loc, index(b, limit), start));
  };
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
  Value zero = index(builder, 0);
  Value input;
  if (!residentInput)
    input = pebble::tensorBuffer(builder, loc, args[0]);
  Value weight = pebble::tensorBuffer(builder, loc, args[1]);
  Value bias = pebble::tensorBuffer(builder, loc, args[2]);
  Value weightSteps = pebble::tensorBuffer(builder, loc, args[5]);
  Value step = tensor::ExtractOp::create(builder, loc, args[4],
                                         ValueRange{zero, zero, zero, zero});
  Value inputMultiplier, outputMultiplier;
  if (floating)
    inputMultiplier = tensor::ExtractOp::create(
        builder, loc, args[3], ValueRange{zero, zero, zero, zero});
  if (fused) {
    SmallVector<Value> indices(
        cast<RankedTensorType>(args[6].getType()).getRank(), zero);
    outputMultiplier =
        tensor::ExtractOp::create(builder, loc, args[6], indices);
  }
  Value output;
  if (!resident)
    output = memref::AllocOp::create(
        builder, loc,
        MemRefType::get(outputType.getShape(), outputType.getElementType()));
  // Host packs: per chunk the weights of one output block in the bank layout
  // of B, the input tile with sixteen channels per group, the bias row, the
  // scales and one tile of results.
  SmallVector<int64_t> groups;
  SmallVector<Value> weightPacks, inputPacks;
  for (auto [start, blocks] : chunks) {
    int64_t outputGroups = pebble::im2colGroups(kernel, blocks, side * side);
    int64_t tiles = kernel * kernel * blocks / outputGroups;
    groups.push_back(outputGroups);
    weightPacks.push_back(memref::AllocOp::create(
        builder, loc,
        MemRefType::get({16 * tiles, outputGroups * 16}, builder.getI8Type())));
    if (!residentInput)
      inputPacks.push_back(memref::AllocOp::create(
          builder, loc,
          floating ? MemRefType::get({inputSide * inputSide * 4, blocks * 4},
                                     builder.getF32Type())
                   : MemRefType::get({inputSide * inputSide, blocks * 16},
                                     builder.getI8Type())));
  }
  Value biasPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value zeroPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value scalePack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getF32Type()));
  Value outPack;
  if (!resident)
    outPack = memref::AllocOp::create(
        builder, loc,
        MemRefType::get({rows, 16}, fused ? Type(builder.getI8Type())
                                          : builder.getF32Type()));
  Value izero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value byteZero = arith::ConstantIntOp::create(builder, loc, 0, 8);
  linalg::FillOp::create(builder, loc, izero, zeroPack);

  // One output block of sixteen channels at `oc`; returns the INT8 result
  // bank of a resident convolution.
  auto outputBlock = [&](OpBuilder &b, Value oc) -> Value {
    Value lanes = extent(b, 16, cout, oc);
    // B: line l of group j is K tile j * tiles + l / 16, input channel l % 16
    // of the tile; tile t is kernel position t / blocks, block t % blocks.
    for (auto [index_, chunkRange] : llvm::enumerate(chunks)) {
      int64_t start = chunkRange.first, blocks = chunkRange.second;
      int64_t tiles = kernel * kernel * blocks / groups[index_];
      Value pack = weightPacks[index_];
      linalg::FillOp::create(b, loc, byteZero, pack);
      loops(
          b, {lanes, index(b, 16 * tiles * groups[index_])},
          [&](OpBuilder &b, ValueRange at) {
            Value lane = at[0], row = at[1];
            Value tile = arith::DivUIOp::create(b, loc, row, index(b, 16));
            Value channel = arith::RemUIOp::create(b, loc, row, index(b, 16));
            Value position =
                arith::DivUIOp::create(b, loc, tile, index(b, blocks));
            Value blockIndex =
                arith::RemUIOp::create(b, loc, tile, index(b, blocks));
            Value ky =
                arith::DivUIOp::create(b, loc, position, index(b, kernel));
            Value kx =
                arith::RemUIOp::create(b, loc, position, index(b, kernel));
            Value o = arith::AddIOp::create(b, loc, oc, lane);
            Value inputChannel;
            if (depthwise)
              inputChannel = arith::AddIOp::create(b, loc, oc, channel);
            else
              inputChannel = arith::AddIOp::create(
                  b, loc,
                  arith::MulIOp::create(b, loc,
                                        arith::AddIOp::create(b, loc,
                                                              blockIndex,
                                                              index(b, start)),
                                        index(b, 16)),
                  channel);
            Value present =
                depthwise
                    ? arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                            channel, lane)
                          .getResult()
                    : arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                            inputChannel, index(b, cin))
                          .getResult();
            auto valid = scf::IfOp::create(b, loc, present, false);
            b.setInsertionPointToStart(&valid.getThenRegion().front());
            Value value = memref::LoadOp::create(
                b, loc, weight,
                ValueRange{o, depthwise ? index(b, 0) : inputChannel, ky, kx});
            Value line =
                arith::RemUIOp::create(b, loc, row, index(b, 16 * tiles));
            Value group =
                arith::DivUIOp::create(b, loc, row, index(b, 16 * tiles));
            memref::StoreOp::create(
                b, loc, value, pack,
                ValueRange{line, arith::AddIOp::create(
                                     b, loc,
                                     arith::MulIOp::create(b, loc, group,
                                                           index(b, 16)),
                                     lane)});
          });
    }
    linalg::FillOp::create(b, loc, izero, biasPack);
    linalg::FillOp::create(
        b, loc,
        ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1))},
        scalePack);
    loops(b, {lanes}, [&](OpBuilder &b, ValueRange at) {
      Value o = arith::AddIOp::create(b, loc, oc, at[0]);
      memref::StoreOp::create(b, loc, memref::LoadOp::create(b, loc, bias, o),
                              biasPack, at[0]);
      Value scale = arith::MulFOp::create(
          b, loc, step,
          memref::LoadOp::create(
              b, loc, weightSteps,
              ValueRange{o, index(b, 0), index(b, 0), index(b, 0)}));
      if (fused)
        scale = arith::MulFOp::create(b, loc, scale, outputMultiplier);
      memref::StoreOp::create(b, loc, scale, scalePack, at[0]);
    });
    Value scales = BankAllocOp::create(b, loc, i64, 1, 1);
    scales = BankMvinOp::create(b, loc, i64, scalePack, scales, 4, 1);
    Value result;
    pebble::repeat(
        b, loc, 0, height, side, false, {},
        [&](OpBuilder &b, Value y, ValueRange) -> SmallVector<Value> {
          pebble::repeat(
              b, loc, 0, width, side, false, {},
              [&](OpBuilder &b, Value x, ValueRange) -> SmallVector<Value> {
                Value acc;
                for (auto [index_, chunkRange] : llvm::enumerate(chunks)) {
                  int64_t start = chunkRange.first, blocks = chunkRange.second;
                  int64_t outputGroups = groups[index_];
                  int64_t tiles = kernel * kernel * blocks / outputGroups;
                  Value source = residentInput;
                  if (!residentInput) {
                    // The input tile of this chunk; the halo outside the
                    // input stays zero.
                    Value pack = inputPacks[index_];
                    Value first =
                        depthwise
                            ? arith::DivUIOp::create(b, loc, oc, index(b, 16))
                                  .getResult()
                            : index(b, start);
                    linalg::FillOp::create(
                        b, loc,
                        ValueRange{floating ? arith::ConstantOp::create(
                                                  b, loc, b.getF32FloatAttr(0))
                                                  .getResult()
                                            : byteZero},
                        pack);
                    loops(
                        b,
                        {index(b, inputSide), index(b, inputSide),
                         index(b, blocks * 16)},
                        [&](OpBuilder &b, ValueRange at) {
                          Value iy = arith::SubIOp::create(
                              b, loc,
                              arith::AddIOp::create(
                                  b, loc,
                                  arith::MulIOp::create(b, loc, y,
                                                        index(b, stride)),
                                  at[0]),
                              index(b, padding[0]));
                          Value ix = arith::SubIOp::create(
                              b, loc,
                              arith::AddIOp::create(
                                  b, loc,
                                  arith::MulIOp::create(b, loc, x,
                                                        index(b, stride)),
                                  at[1]),
                              index(b, padding[2]));
                          Value channel = arith::AddIOp::create(
                              b, loc,
                              arith::MulIOp::create(b, loc, first,
                                                    index(b, 16)),
                              at[2]);
                          auto inRange = [&](Value value, int64_t limit) {
                            return arith::AndIOp::create(
                                b, loc,
                                arith::CmpIOp::create(b, loc,
                                                      arith::CmpIPredicate::sge,
                                                      value, index(b, 0)),
                                arith::CmpIOp::create(b, loc,
                                                      arith::CmpIPredicate::slt,
                                                      value, index(b, limit)));
                          };
                          Value inside = arith::AndIOp::create(
                              b, loc,
                              arith::AndIOp::create(
                                  b, loc, inRange(iy, inputType.getDimSize(1)),
                                  inRange(ix, inputType.getDimSize(2))),
                              inRange(channel, cin));
                          auto valid = scf::IfOp::create(b, loc, inside, false);
                          b.setInsertionPointToStart(
                              &valid.getThenRegion().front());
                          Value value = memref::LoadOp::create(
                              b, loc, input,
                              ValueRange{index(b, 0), iy, ix, channel});
                          Value pixel = arith::AddIOp::create(
                              b, loc,
                              arith::MulIOp::create(b, loc, at[0],
                                                    index(b, inputSide)),
                              at[1]);
                          // FP32: line 4 * pixel + c / 4 of group c / 16.
                          Value line = pixel, lane = at[2];
                          if (floating) {
                            Value block = arith::DivUIOp::create(b, loc, at[2],
                                                                 index(b, 16));
                            Value inBlock = arith::RemUIOp::create(
                                b, loc, at[2], index(b, 16));
                            line = arith::AddIOp::create(
                                b, loc,
                                arith::MulIOp::create(b, loc, pixel,
                                                      index(b, 4)),
                                arith::DivUIOp::create(b, loc, inBlock,
                                                       index(b, 4)));
                            lane = arith::AddIOp::create(
                                b, loc,
                                arith::MulIOp::create(b, loc, block,
                                                      index(b, 4)),
                                arith::RemUIOp::create(b, loc, inBlock,
                                                       index(b, 4)));
                          }
                          memref::StoreOp::create(b, loc, value, pack,
                                                  ValueRange{line, lane});
                        });
                    int64_t pixels = inputSide * inputSide;
                    source = BankAllocOp::create(b, loc, i64, 1, blocks);
                    if (floating) {
                      Value values =
                          BankAllocOp::create(b, loc, i64, 1, blocks);
                      values = BankMvinOp::create(b, loc, i64, pack, values,
                                                  pixels * 4, 1);
                      source = ::buddy::toint8_ball::BankFp32ToInt8Op::create(
                          b, loc, i64, values, source, inputMultiplier,
                          pixels * 4);
                      BankReleaseOp::create(b, loc, values);
                    } else {
                      source = BankMvinOp::create(b, loc, i64, pack, source,
                                                  pixels, 1);
                    }
                  }
                  Value a = BankAllocOp::create(b, loc, i64, 1, outputGroups);
                  a = ::buddy::im2col_ball::BankIm2colI8Op::create(
                      b, loc, i64, source, a, inputSide, kernel, stride,
                      im2colPadding, pebble::tileRows());
                  if (!residentInput)
                    BankReleaseOp::create(b, loc, source);
                  Value w = BankAllocOp::create(b, loc, i64, 1, outputGroups);
                  w = BankMvinOp::create(b, loc, i64, weightPacks[index_], w,
                                         16 * tiles, 1);
                  // The first chunk starts from the bias, the others from
                  // zero and add to the accumulator.
                  Value partial = BankAllocOp::create(b, loc, i64, 1, 1);
                  partial = BankMvinOp::create(
                      b, loc, i64, acc ? zeroPack : biasPack, partial, 4, 1);
                  partial = ::buddy::smatmul_ball::BankMatmulI8Op::create(
                      b, loc, i64, a, w, partial, rows, 16 * tiles);
                  BankReleaseOp::create(b, loc, a);
                  BankReleaseOp::create(b, loc, w);
                  if (!acc) {
                    acc = partial;
                    continue;
                  }
                  Value sum = BankAllocOp::create(b, loc, i64, 1, 1);
                  sum = ::buddy::matadd_ball::BankAddI32Op::create(
                      b, loc, i64, acc, partial, sum, rows * 4);
                  BankReleaseOp::create(b, loc, acc);
                  BankReleaseOp::create(b, loc, partial);
                  acc = sum;
                }
                Value converted = BankAllocOp::create(b, loc, i64, 1, 1);
                converted =
                    fused
                        ? Value(::buddy::toint8_ball::BankInt32ToInt8Op::create(
                              b, loc, i64, acc, scales, converted, rows, relu))
                        : Value(::buddy::int2fp_ball::BankInt32ToFp32Op::create(
                              b, loc, i64, acc, scales, converted, rows, relu));
                BankReleaseOp::create(b, loc, acc);
                if (pool > 1) {
                  Value pooledBank = BankAllocOp::create(b, loc, i64, 1, 1);
                  pooledBank = ::buddy::maxpool_ball::BankMaxpoolI8Op::create(
                      b, loc, i64, converted, pooledBank, side, pool, pool, 0);
                  BankReleaseOp::create(b, loc, converted);
                  converted = pooledBank;
                }
                if (resident) {
                  result = converted;
                  return {};
                }
                converted = BankMvoutOp::create(
                    b, loc, i64, outPack, converted,
                    fused ? (pool > 1 ? pooled * pooled : rows) : rows * 4, 1);
                BankReleaseOp::create(b, loc, converted);
                // The pixels of the tile inside the output.
                Value py = arith::DivUIOp::create(b, loc, y, index(b, pool));
                Value px = arith::DivUIOp::create(b, loc, x, index(b, pool));
                loops(
                    b,
                    {extent(b, pooled, height / pool, py),
                     extent(b, pooled, width / pool, px), lanes},
                    [&](OpBuilder &b, ValueRange at) {
                      Value pixel = arith::AddIOp::create(
                          b, loc,
                          arith::MulIOp::create(b, loc, at[0],
                                                index(b, pooled)),
                          at[1]);
                      Value value = memref::LoadOp::create(
                          b, loc, outPack, ValueRange{pixel, at[2]});
                      memref::StoreOp::create(
                          b, loc, value, output,
                          ValueRange{index(b, 0),
                                     arith::AddIOp::create(b, loc, at[0], py),
                                     arith::AddIOp::create(b, loc, at[1], px),
                                     arith::AddIOp::create(b, loc, at[2], oc)});
                    });
                return {};
              });
          return {};
        });
    BankReleaseOp::create(b, loc, scales);
    return result;
  };

  if (resident) {
    // Output blocks append their groups to one bank.
    for (int64_t oc = 0; oc < cout; oc += 16) {
      Value block = outputBlock(builder, index(builder, oc));
      image = image ? BankTransferOp::create(builder, loc, i64, block, image)
                          .getResult()
                    : block;
    }
  } else {
    pebble::repeat(builder, loc, 0, cout, 16, false, {},
                   [&](OpBuilder &b, Value oc, ValueRange) {
                     outputBlock(b, oc);
                     return SmallVector<Value>{};
                   });
    tensorOutput = bufferization::ToTensorOp::create(builder, loc, outputType,
                                                     output, true, true);
  }
  if (residentInput)
    BankReleaseOp::create(builder, loc, residentInput);
  return success();
}
} // namespace

LogicalResult pebble::lowerSpatialOps(func::FuncOp function,
                                      DenseMap<Value, Value> &residentVectors,
                                      SmallVector<Operation *> &consumed) {
  SmallVector<Computation> ops;
  function.walk([&](Computation op) {
    if (op.getKind() == "conv2d_i8" || op.getKind() == "depthwise_conv2d_i8" ||
        op.getKind() == "flatten_nchw")
      ops.push_back(op);
  });
  DenseMap<Value, Value> images;
  for (auto op : ops) {
    Value input = images.lookup(op.getInputs()[0]);
    if (op.getKind() == "flatten_nchw") {
      // A 4x4 map of sixteen channels in channel-major order: the transposed
      // 16x16 block, one K block of sixteen rows for a one-row matmul.
      OpBuilder b(op);
      Value vector = BankAllocOp::create(b, op.getLoc(), b.getI64Type(), 1, 1);
      vector = ::buddy::transpose_ball::BankTransposeI8Op::create(
          b, op.getLoc(), b.getI64Type(), input, vector, 16, 16);
      BankReleaseOp::create(b, op.getLoc(), input);
      residentVectors[op.getResult(0)] = vector;
      consumed.push_back(op);
      continue;
    }
    Value image, tensor;
    if (failed(convolution(op, input, image, tensor)))
      return failure();
    if (image) {
      images[op.getResult(0)] = image;
      consumed.push_back(op);
      pebble::observeResident(op, image);
    } else {
      op.getResult(0).replaceAllUsesWith(tensor);
      op.erase();
    }
  }
  return success();
}
