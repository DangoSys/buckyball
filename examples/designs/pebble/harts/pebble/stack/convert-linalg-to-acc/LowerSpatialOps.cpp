//===- LowerSpatialOps.cpp - Convolutions in bank tiles -------------------===//
//
// A convolution computes sixteen output channels per matmul_i8 over a tile of
// output pixels. The input tile, sixteen channels per group, goes through
// im2col_i8 into the A operand; B holds the weights in the same K order.
// An FP32 input converts to INT8 with one multiplier per input channel, or
// one for the whole input. Input channels beyond one call form chunks
// (convChunk) whose partial products add up with add_i32. A fused convolution
// requantizes for its consumer and pools; when the consumer reads the whole
// result, the result stays in banks, one group per channel block.
//
//===----------------------------------------------------------------------===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "fixedOps.h"
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
using namespace ::buddy::acc;
using Computation = ::buddy::acc::CustomOp;

namespace {
// The weight parameter of a convolution stored offline in the B layout (the
// recipe's frontend): [F / 16, chunks, kernel, kernel, 16 * chunk, 16] as
// [F / 16, rest], zero-padded, the sixteen output channels of a K line last;
// the graph reads it through expand_shape, transpose (0, 5, 1, 4, 2, 3),
// collapse_shape and zero-offset slices to [F, C, kernel, kernel]. Line l of
// a chunk is K line l of its matmul, so every group of B is a DMA in place.
// Null when `value` is not read so; `chunk` gives the channel blocks of a
// chunk.
Value storedWeight(Value value, int64_t &chunk) {
  while (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (!llvm::all_of(slice.getStaticOffsets(),
                      [](int64_t offset) { return offset == 0; }) ||
        !llvm::all_of(slice.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return {};
    value = slice.getSource();
  }
  auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>();
  auto transpose = collapse
                       ? collapse.getSrc().getDefiningOp<linalg::TransposeOp>()
                       : linalg::TransposeOp();
  auto expand =
      transpose ? transpose.getInput().getDefiningOp<tensor::ExpandShapeOp>()
                : tensor::ExpandShapeOp();
  if (!expand ||
      transpose.getPermutation() != ArrayRef<int64_t>{0, 5, 1, 4, 2, 3})
    return {};
  chunk = expand.getResultType().getDimSize(4) / 16;
  return expand.getSrc();
}

// The recipe stores sixteen depthwise channels as diagonal 16x16 blocks.
// Its logical FCHW view selects every seventeenth byte of each block.
Value storedDepthwiseWeight(Value value) {
  if (auto trim = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (!llvm::all_of(trim.getStaticOffsets(),
                      [](int64_t n) { return n == 0; }) ||
        !llvm::all_of(trim.getStaticStrides(),
                      [](int64_t n) { return n == 1; }))
      return {};
    value = trim.getSource();
  }
  auto expand = value.getDefiningOp<tensor::ExpandShapeOp>();
  auto collapse = expand
                      ? expand.getSrc().getDefiningOp<tensor::CollapseShapeOp>()
                      : tensor::CollapseShapeOp();
  auto transpose = collapse
                       ? collapse.getSrc().getDefiningOp<linalg::TransposeOp>()
                       : linalg::TransposeOp();
  auto diagonal =
      transpose ? transpose.getInput().getDefiningOp<tensor::ExtractSliceOp>()
                : tensor::ExtractSliceOp();
  if (!diagonal ||
      transpose.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2} ||
      diagonal.getStaticStrides() != ArrayRef<int64_t>{1, 1, 1, 17} ||
      !llvm::all_of(diagonal.getStaticOffsets(),
                    [](int64_t n) { return n == 0; }))
    return {};
  auto blocks = diagonal.getSource().getDefiningOp<tensor::ExpandShapeOp>();
  return blocks ? blocks.getSrc() : Value{};
}

// Lowers `op`. A resident input is a bank with one group per input channel
// block holding the whole input map. With bbacc.resident the INT8 result is
// returned in `image`, one group per output channel block; otherwise in
// `tensorOutput`.
LogicalResult convolution(Computation op, Value residentInput, Value &image,
                          Value &tensorOutput) {
  bool depthwise = op.getKind().starts_with("depthwise_conv2d_i8");
  bool accumulator = op.getKind().ends_with("_acc");
  bool fixed = op->hasAttr("fixed_fraction");
  bool packedDepthwise = depthwise && accumulator;
  bool fused = op.getInputs().size() == 7;
  bool resident = op->hasAttr("bbacc.resident");
  auto args = op.getInputs();
  auto inputType = cast<RankedTensorType>(args[0].getType());
  auto weightType = cast<RankedTensorType>(args[1].getType());
  auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
  auto strides = cast<DenseI64ArrayAttr>(op->getAttr("strides")).asArrayRef();
  auto padding = cast<DenseI64ArrayAttr>(op->getAttr("padding")).asArrayRef();
  if (args.size() != (accumulator ? (fixed ? 5 : 3)
                      : fused     ? 7
                                  : 6) ||
      !inputType.hasStaticShape() || inputType.getRank() != 4 ||
      !weightType.hasStaticShape() || weightType.getRank() != 4 ||
      inputType.getDimSize(0) != 1 ||
      (accumulator && !inputType.getElementType().isInteger(8)) ||
      !weightType.getElementType().isInteger(8) ||
      !cast<RankedTensorType>(args[2].getType()).getElementType().isInteger(32))
    return op.emitOpError("requires static NHWC/FCHW convolution with INT32 "
                          "bias");
  bool floating = inputType.getElementType().isF32();
  bool wordInput = floating || inputType.getElementType().isInteger(32);
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
                           blocksIn, depthwise, wordInput);
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
  // Preserve spatial strides and channel-slice offsets for direct DMA.
  if (!residentInput)
    input = bufferization::ToBufferOp::create(
        builder, loc,
        MemRefType::get(inputType.getShape(), inputType.getElementType(),
                        StridedLayoutAttr::get(
                            builder.getContext(), ShapedType::kDynamic,
                            SmallVector<int64_t>(4, ShapedType::kDynamic))),
        args[0], true);
  // Depthwise B is diagonal, packed on the host from the [C, 1, kernel,
  // kernel] weight; any other is the stored weight in place.
  Value weight, stored, storedParameter;
  int64_t storedChunk = 0;
  if (depthwise && !packedDepthwise) {
    weight = pebble::tensorBuffer(builder, loc, args[1]);
  } else {
    Value parameter = packedDepthwise ? storedDepthwiseWeight(args[1])
                                      : storedWeight(args[1], storedChunk);
    if (!parameter || (!packedDepthwise && storedChunk != chunk) ||
        cast<RankedTensorType>(parameter.getType()).getShape() !=
            ArrayRef<int64_t>{(cout + 15) / 16,
                              (packedDepthwise ? 1 : blocksIn) * kernel *
                                  kernel * 256})
      return op.emitOpError("requires the weight stored in the pebble B "
                            "layout in chunks of ")
             << chunk << " channel blocks";
    storedParameter = parameter;
    stored = pebble::tensorBuffer(builder, loc, parameter);
  }
  Value bias = pebble::tensorBuffer(builder, loc, args[2]);
  if (op->hasAttr("bbacc.matrix") && !fixed) {
    // A unit-stride 1x1 convolution is a matrix product over NHWC pixels. Its
    // stored B payload already has the linear schedule's [N/16, K, 16] order.
    Value matrixInput = memref::CollapseShapeOp::create(
        builder, loc, input, ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}});
    Value packed = tensor::ExpandShapeOp::create(
        builder, loc,
        RankedTensorType::get({cout / 16, cin, 16}, builder.getI8Type()),
        storedParameter, ArrayRef<ReassociationIndices>{{0}, {1, 2}});
    Value matrixBias = tensor::ExpandShapeOp::create(
        builder, loc, RankedTensorType::get({1, cout}, builder.getI32Type()),
        args[2], ArrayRef<ReassociationIndices>{{0, 1}});
    Value output = memref::AllocOp::create(
        builder, loc,
        MemRefType::get(outputType.getShape(), outputType.getElementType()));
    Value matrixOutput = memref::CollapseShapeOp::create(
        builder, loc, output, ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}});
    pebble::linearTiles(builder, loc, matrixInput, Value(), 16, Value(),
                        pebble::tensorBuffer(builder, loc, packed),
                        pebble::tensorBuffer(builder, loc, matrixBias), Value(),
                        false, matrixOutput);
    tensorOutput = bufferization::ToTensorOp::create(builder, loc, outputType,
                                                     output, true, true);
    return success();
  }
  Value fixedMultipliers, fixedShifts;
  if (fixed) {
    fixedMultipliers = pebble::tensorBuffer(builder, loc, args[3]);
    fixedShifts = pebble::tensorBuffer(builder, loc, args[4]);
  }
  Value weightSteps, step;
  if (!accumulator) {
    weightSteps = pebble::tensorBuffer(builder, loc, args[5]);
    step = tensor::ExtractOp::create(builder, loc, args[4],
                                     ValueRange{zero, zero, zero, zero});
  }
  Value inputMultipliers, outputMultipliers;
  bool perChannel = false;
  if (wordInput) {
    int64_t count = cast<RankedTensorType>(args[3].getType()).getDimSize(3);
    if (count != 1 && count != cin)
      return op.emitOpError("requires one input multiplier per input channel "
                            "or one for the whole input");
    perChannel = count == cin;
    Value parameter = args[3];
    inputMultipliers = pebble::tensorBuffer(builder, loc, parameter);
  }
  // A fused convolution converts its result with the input multipliers of
  // its consumer: one per output channel, or one for the whole result.
  bool perOutputChannel = false;
  if (fused) {
    auto type = cast<RankedTensorType>(args[6].getType());
    int64_t count = type.getDimSize(type.getRank() - 1);
    if (count != 1 && count != cout)
      return op.emitOpError("requires one output multiplier per output "
                            "channel or one for the whole result");
    perOutputChannel = count == cout;
    outputMultipliers = pebble::tensorBuffer(builder, loc, args[6]);
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
  SmallVector<Value> weightPacks, multiplierPacks;
  for (auto [start, blocks] : chunks) {
    int64_t outputGroups = pebble::im2colGroups(kernel, blocks, side * side);
    int64_t tiles = kernel * kernel * blocks / outputGroups;
    groups.push_back(outputGroups);
    if (depthwise && !packedDepthwise)
      weightPacks.push_back(memref::AllocOp::create(
          builder, loc,
          MemRefType::get({16 * tiles, outputGroups * 16},
                          builder.getI8Type())));
    // The fp32_to_int8 multipliers of the chunk: rows 4g..4g+3 of one group
    // for channel block g.
    if (wordInput && !accumulator)
      multiplierPacks.push_back(memref::AllocOp::create(
          builder, loc,
          MemRefType::get({blocks * 4, 4}, builder.getF32Type())));
  }
  Value biasPack;
  if (!accumulator)
    biasPack = memref::AllocOp::create(
        builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value zeroPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  // At the largest legal K grouping, A/B and the accumulator fill the banks.
  // Load the FP32 scales only after the weight banks have been released.
  bool deferScales = !accumulator && 2 * *llvm::max_element(groups) + 2 >
                                         target::getTarget().bankNum - 1;
  Value scalePack;
  if (!accumulator)
    scalePack = memref::AllocOp::create(
        builder, loc, MemRefType::get({16}, builder.getF32Type()));
  Value izero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value byteZero = arith::ConstantIntOp::create(builder, loc, 0, 8);
  linalg::FillOp::create(builder, loc, izero, zeroPack);

  auto prepareOutput = [&](OpBuilder &b, Value oc) -> Value {
    if (fixed) {
      Value parameters = BankAllocOp::create(b, loc, i64, 1, 1);
      for (int part = 0; part < 2; ++part) {
        Value memory = memref::SubViewOp::create(
            b, loc, part ? fixedShifts : fixedMultipliers,
            ArrayRef<OpFoldResult>{oc},
            ArrayRef<OpFoldResult>{b.getIndexAttr(16)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        parameters = BankMvinStridedOp::create(
            b, loc, i64, memory, parameters, index(b, part * 4), index(b, 4),
            index(b, 16), 0, 1, index(b, 16),
            arith::ConstantIndexOp::create(b, loc, 0));
      }
      return parameters;
    }
    if (accumulator)
      return {};
    Value lanes = extent(b, 16, cout, oc);
    // B: line l of group j is K tile j * tiles + l / 16, input channel l % 16
    // of the tile; tile t is kernel position t / blocks, block t % blocks.
    for (auto [index_, chunkRange] : llvm::enumerate(chunks)) {
      if (!depthwise || packedDepthwise)
        break;
      int64_t blocks = chunkRange.second;
      int64_t tiles = kernel * kernel * blocks / groups[index_];
      Value pack = weightPacks[index_];
      linalg::FillOp::create(b, loc, byteZero, pack);
      // Only the diagonal is nonzero; do not scan the other 15 lanes.
      loops(b, {lanes, index(b, kernel * kernel)},
            [&](OpBuilder &b, ValueRange at) {
              Value lane = at[0], position = at[1];
              Value ky =
                  arith::DivUIOp::create(b, loc, position, index(b, kernel));
              Value kx =
                  arith::RemUIOp::create(b, loc, position, index(b, kernel));
              Value o = arith::AddIOp::create(b, loc, oc, lane);
              Value value = memref::LoadOp::create(b, loc, weight,
                                                   ValueRange{o, zero, ky, kx});
              Value row = arith::AddIOp::create(
                  b, loc, arith::MulIOp::create(b, loc, position, index(b, 16)),
                  lane);
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
    if (cout % 16)
      linalg::FillOp::create(b, loc, izero, biasPack);
    linalg::FillOp::create(
        b, loc,
        ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1))},
        scalePack);
    loops(b, {lanes}, [&](OpBuilder &b, ValueRange at) {
      Value o = arith::AddIOp::create(b, loc, oc, at[0]);
      if (cout % 16)
        memref::StoreOp::create(b, loc, memref::LoadOp::create(b, loc, bias, o),
                                biasPack, at[0]);
      Value scale = arith::MulFOp::create(
          b, loc, step,
          memref::LoadOp::create(
              b, loc, weightSteps,
              ValueRange{o, index(b, 0), index(b, 0), index(b, 0)}));
      if (fused) {
        SmallVector<Value> indices(
            cast<MemRefType>(outputMultipliers.getType()).getRank(),
            index(b, 0));
        if (perOutputChannel)
          indices.back() = o;
        scale = arith::MulFOp::create(
            b, loc, scale,
            memref::LoadOp::create(b, loc, outputMultipliers, indices));
      }
      memref::StoreOp::create(b, loc, scale, scalePack, at[0]);
    });
    if (deferScales)
      return {};
    Value scales = BankAllocOp::create(b, loc, i64, 1, 1);
    scales = BankMvinOp::create(b, loc, i64, scalePack, scales, 4, 1);
    return scales;
  };

  auto prepareMultipliers = [&](OpBuilder &b, Value oc) {
    // Multiplier of channel c of a chunk: row c / 4, lane c % 4. A padding
    // channel reads zero inputs; any positive multiplier converts them.
    if (wordInput && !accumulator)
      for (auto [index_, chunkRange] : llvm::enumerate(chunks)) {
        int64_t start = chunkRange.first, blocks = chunkRange.second;
        Value pack = multiplierPacks[index_];
        linalg::FillOp::create(
            b, loc,
            ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1))},
            pack);
        Value first =
            depthwise
                ? arith::DivUIOp::create(b, loc, oc, index(b, 16)).getResult()
                : index(b, start);
        loops(b, {index(b, blocks * 16)}, [&](OpBuilder &b, ValueRange at) {
          Value channel = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, first, index(b, 16)),
              at[0]);
          auto valid = scf::IfOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, channel,
                                    index(b, cin)),
              false);
          b.setInsertionPointToStart(&valid.getThenRegion().front());
          Value value = memref::LoadOp::create(
              b, loc, inputMultipliers,
              ValueRange{zero, zero, zero, perChannel ? channel : zero});
          memref::StoreOp::create(
              b, loc, value, pack,
              ValueRange{arith::DivUIOp::create(b, loc, at[0], index(b, 4)),
                         arith::RemUIOp::create(b, loc, at[0], index(b, 4))});
        });
      }
  };

  auto inputBank = [&](OpBuilder &b, size_t index_, Value y, Value x,
                       Value oc) -> Value {
    int64_t blocks = chunks[index_].second;
    int64_t outputGroups = groups[index_];
    Value source = residentInput;
    if (!residentInput) {
      int64_t pixels = inputSide * inputSide;
      int64_t lines = wordInput ? 4 : 1;
      int64_t elementBytes = wordInput ? 4 : 1;
      Value values = BankAllocOp::create(b, loc, i64, 1, blocks);
      values = BankClearOp::create(b, loc, i64, values, 1, blocks);
      Value iy0 = arith::SubIOp::create(
          b, loc, arith::MulIOp::create(b, loc, y, index(b, stride)),
          index(b, padding[0]));
      Value ix0 = arith::SubIOp::create(
          b, loc, arith::MulIOp::create(b, loc, x, index(b, stride)),
          index(b, padding[2]));
      Value yBegin = arith::MaxSIOp::create(
          b, loc, zero, arith::SubIOp::create(b, loc, zero, iy0));
      Value xBegin = arith::MaxSIOp::create(
          b, loc, zero, arith::SubIOp::create(b, loc, zero, ix0));
      Value xEnd = extent(b, inputSide, inputType.getDimSize(2), ix0);
      Value count = arith::SubIOp::create(b, loc, xEnd, xBegin);
      Value yEnd = arith::SelectOp::create(
          b, loc,
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, count, zero),
          extent(b, inputSide, inputType.getDimSize(1), iy0), yBegin);
      Value memoryStride = arith::MulIOp::create(
          b, loc,
          memref::ExtractStridedMetadataOp::create(b, loc, input)
              .getStrides()[2],
          index(b, elementBytes));
      Value firstChannel = depthwise ? oc : index(b, chunks[index_].first * 16);
      auto spatial = scf::ForOp::create(
          b, loc, yBegin, yEnd, index(b, 1), ValueRange{values},
          [&](OpBuilder &b, Location loc, Value iy, ValueRange carried) {
            Value bank = carried[0];
            for (int64_t group = 0; group < blocks; ++group) {
              Value channel = arith::AddIOp::create(b, loc, firstChannel,
                                                    index(b, group * 16));
              Value channels = extent(b, 16, cin, channel);
              if (elementBytes == 4 && cin % 8 == 0) {
                Value blockLog = arith::SelectOp::create(
                    b, loc,
                    arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                          channels, index(b, 16)),
                    index(b, 2), index(b, 1));
                Value memory = memref::SubViewOp::create(
                    b, loc, input,
                    ArrayRef<OpFoldResult>{
                        zero,
                        arith::AddIOp::create(b, loc, iy0, iy).getResult(),
                        arith::AddIOp::create(b, loc, ix0, xBegin).getResult(),
                        channel},
                    SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
                    SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
                Value first = arith::MulIOp::create(
                    b, loc, index(b, 4),
                    arith::AddIOp::create(
                        b, loc,
                        arith::MulIOp::create(b, loc, iy, index(b, inputSide)),
                        xBegin));
                bank = BankMvinStridedOp::create(
                    b, loc, i64, memory, bank, first,
                    arith::ShLIOp::create(b, loc, count, blockLog),
                    memoryStride, group, 4, index(b, 16), blockLog);
                continue;
              }
              Value quarters = arith::CeilDivUIOp::create(
                  b, loc, channels, index(b, 16 / elementBytes));
              auto lanes = scf::ForOp::create(
                  b, loc, zero, quarters, index(b, 1), ValueRange{bank},
                  [&](OpBuilder &b, Location loc, Value quarter,
                      ValueRange carried) {
                    Value c = arith::AddIOp::create(
                        b, loc, channel,
                        arith::MulIOp::create(b, loc, quarter,
                                              index(b, 16 / elementBytes)));
                    Value memory = memref::SubViewOp::create(
                        b, loc, input,
                        ArrayRef<OpFoldResult>{
                            zero,
                            arith::AddIOp::create(b, loc, iy0, iy).getResult(),
                            arith::AddIOp::create(b, loc, ix0, xBegin)
                                .getResult(),
                            c},
                        ArrayRef<OpFoldResult>{
                            b.getIndexAttr(1), b.getIndexAttr(1),
                            b.getIndexAttr(1), b.getIndexAttr(1)},
                        ArrayRef<OpFoldResult>{
                            b.getIndexAttr(1), b.getIndexAttr(1),
                            b.getIndexAttr(1), b.getIndexAttr(1)});
                    Value first = arith::AddIOp::create(
                        b, loc, quarter,
                        arith::MulIOp::create(
                            b, loc, index(b, lines),
                            arith::AddIOp::create(
                                b, loc,
                                arith::MulIOp::create(b, loc, iy,
                                                      index(b, inputSide)),
                                xBegin)));
                    Value bytes = arith::MulIOp::create(
                        b, loc, extent(b, 16 / elementBytes, cin, c),
                        index(b, elementBytes));
                    Value loaded = BankMvinStridedOp::create(
                        b, loc, i64, memory, carried[0], first, count,
                        memoryStride, group, lines, bytes,
                        arith::ConstantIndexOp::create(b, loc, 0));
                    scf::YieldOp::create(b, loc, loaded);
                  });
              bank = lanes.getResult(0);
            }
            scf::YieldOp::create(b, loc, bank);
          });
      values = spatial.getResult(0);
      if (wordInput) {
        source = BankAllocOp::create(b, loc, i64, 1, blocks);
        Value multipliers = BankAllocOp::create(b, loc, i64, 1, 1);
        multipliers = BankMvinOp::create(b, loc, i64, multiplierPacks[index_],
                                         multipliers, blocks * 4, 1);
        source =
            floating
                ? Value(::buddy::toint8_ball::BankFp32ToInt8Op::create(
                      b, loc, i64, values, multipliers, source, pixels * 4))
                : Value(::buddy::toint8_ball::BankInt32ToInt8Op::create(
                      b, loc, i64, values, multipliers, source, pixels, false));
        BankReleaseOp::create(b, loc, values);
        BankReleaseOp::create(b, loc, multipliers);
      } else {
        source = values;
      }
    }
    Value a = BankAllocOp::create(b, loc, i64, 1, outputGroups);
    a = ::buddy::im2col_ball::BankIm2colI8Op::create(
        b, loc, i64, source, a, inputSide, kernel, stride, im2colPadding,
        pebble::tileRows());
    if (!residentInput)
      BankReleaseOp::create(b, loc, source);
    return a;
  };

  auto outputTile = [&](OpBuilder &b, Value y, Value x, Value oc, Value scales,
                        Value sharedInput) -> Value {
    Value acc;
    for (auto [index_, chunkRange] : llvm::enumerate(chunks)) {
      int64_t start = chunkRange.first, blocks = chunkRange.second;
      int64_t outputGroups = groups[index_];
      int64_t tiles = kernel * kernel * blocks / outputGroups;
      Value a = sharedInput ? sharedInput : inputBank(b, index_, y, x, oc);
      Value w = BankAllocOp::create(b, loc, i64, 1, outputGroups);
      if (depthwise && !packedDepthwise)
        w = BankMvinOp::create(b, loc, i64, weightPacks[index_], w, 16 * tiles,
                               1);
      // Group j: K lines j * 16 * tiles on of the chunk, whose
      // lines start at column start * kernel * kernel * 256 of
      // the stored row of the output block.
      for (int64_t group = 0;
           (!depthwise || packedDepthwise) && group < outputGroups; ++group)
        w = BankMvinGroupOp::create(
            b, loc, i64,
            memref::SubViewOp::create(
                b, loc, stored,
                ArrayRef<OpFoldResult>{
                    arith::DivUIOp::create(b, loc, oc, index(b, 16))
                        .getResult(),
                    b.getIndexAttr(start * kernel * kernel * 256 +
                                   group * tiles * 256)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                       b.getIndexAttr(tiles * 256)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)}),
            w, group, 16 * tiles, 1);
      // The first chunk starts from the bias, the others from
      // zero and add to the accumulator.
      Value partial = BankAllocOp::create(b, loc, i64, 1, 1);
      if (acc || !accumulator) {
        Value initial = acc ? zeroPack : biasPack;
        if (!acc && cout % 16 == 0)
          initial = memref::SubViewOp::create(
              b, loc, bias, ArrayRef<OpFoldResult>{oc},
              ArrayRef<OpFoldResult>{b.getIndexAttr(16)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        partial = BankMvinOp::create(b, loc, i64, initial, partial, 4, 1);
      } else {
        // Read the bias parameter directly; only a tail needs zero padding.
        if (cout % 16)
          partial = BankMvinOp::create(b, loc, i64, zeroPack, partial, 4, 1);
        Value lanes = extent(b, 16, cout, oc);
        Value biasRows = arith::CeilDivUIOp::create(b, loc, lanes, index(b, 4));
        if (cout % 4 == 0) {
          Value memory = memref::SubViewOp::create(
              b, loc, bias, ArrayRef<OpFoldResult>{oc},
              ArrayRef<OpFoldResult>{lanes},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          partial = BankMvinStridedOp::create(
              b, loc, i64, memory, partial, zero, biasRows, index(b, 16), 0, 1,
              index(b, 16), arith::ConstantIndexOp::create(b, loc, 0));
        } else {
          auto load = scf::ForOp::create(
              b, loc, zero, biasRows, index(b, 1), ValueRange{partial},
              [&](OpBuilder &b, Location, Value row, ValueRange carried) {
                Value channel = arith::AddIOp::create(
                    b, loc, oc,
                    arith::MulIOp::create(b, loc, row, index(b, 4)));
                Value memory = memref::SubViewOp::create(
                    b, loc, bias, ArrayRef<OpFoldResult>{channel},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                Value bytes = arith::MulIOp::create(
                    b, loc, extent(b, 4, cout, channel), index(b, 4));
                Value loaded = BankMvinStridedOp::create(
                    b, loc, i64, memory, carried[0], row, index(b, 1),
                    index(b, 16), 0, 1, bytes,
                    arith::ConstantIndexOp::create(b, loc, 0));
                scf::YieldOp::create(b, loc, loaded);
              });
          partial = load.getResult(0);
        }
      }
      partial = ::buddy::smatmul_ball::BankMatmulI8Op::create(
          b, loc, i64, a, w, partial, rows, 16 * tiles);
      if (!sharedInput)
        BankReleaseOp::create(b, loc, a);
      BankReleaseOp::create(b, loc, w);
      if (!acc) {
        acc = partial;
        continue;
      }
      Value sum = BankAllocOp::create(b, loc, i64, 1, 1);
      sum = ::buddy::matadd_ball::BankAddI32Op::create(b, loc, i64, acc,
                                                       partial, sum, rows * 4);
      BankReleaseOp::create(b, loc, acc);
      BankReleaseOp::create(b, loc, partial);
      acc = sum;
    }
    Value lanes = extent(b, 16, cout, oc);
    Value converted = acc;
    if (deferScales) {
      scales = BankAllocOp::create(b, loc, i64, 1, 1);
      scales = BankMvinOp::create(b, loc, i64, scalePack, scales, 4, 1);
    }
    if (fixed) {
      converted = BankAllocOp::create(b, loc, i64, 1, 1);
      converted = ::buddy::fixed_ball::BankFixedOp::create(
          b, loc, i64, acc, scales, converted, side * side * 4, 0,
          cast<IntegerAttr>(op->getAttr("fixed_fraction")).getInt(), false,
          false);
      if (op->hasAttr("fixed_relu")) {
        Value requantized = converted;
        converted = ::buddy::fixed_ball::BankFixedOp::create(
            b, loc, i64, requantized, scales, acc, side * side * 4, 4,
            cast<IntegerAttr>(op->getAttr("fixed_fraction")).getInt(), false,
            false);
        BankReleaseOp::create(b, loc, requantized);
      } else {
        BankReleaseOp::create(b, loc, acc);
      }
    } else if (!accumulator) {
      converted = BankAllocOp::create(b, loc, i64, 1, 1);
      converted = fused ? Value(::buddy::toint8_ball::BankInt32ToInt8Op::create(
                              b, loc, i64, acc, scales, converted, rows, relu))
                        : Value(::buddy::int2fp_ball::BankInt32ToFp32Op::create(
                              b, loc, i64, acc, scales, converted, rows, relu));
      BankReleaseOp::create(b, loc, acc);
    }
    if (deferScales)
      BankReleaseOp::create(b, loc, scales);
    if (pool > 1) {
      Value pooledBank = BankAllocOp::create(b, loc, i64, 1, 1);
      pooledBank = ::buddy::maxpool_ball::BankMaxpoolI8Op::create(
          b, loc, i64, converted, pooledBank, side, pool, pool, 0);
      BankReleaseOp::create(b, loc, converted);
      converted = pooledBank;
    }
    if (resident) {
      return converted;
    }
    Value py = arith::DivUIOp::create(b, loc, y, index(b, pool));
    Value px = arith::DivUIOp::create(b, loc, x, index(b, pool));
    int64_t elementBytes = fused ? 1 : 4;
    int64_t lines = fused ? 1 : 4;
    Value quarters =
        arith::CeilDivUIOp::create(b, loc, lanes, index(b, 16 / elementBytes));
    if (elementBytes == 4 && cout % 8 == 0) {
      Value blockLog = arith::SelectOp::create(
          b, loc,
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lanes,
                                index(b, 16)),
          index(b, 2), index(b, 1));
      loops(b, {extent(b, pooled, height / pool, py)},
            [&](OpBuilder &b, ValueRange at) {
              Value memory = memref::SubViewOp::create(
                  b, loc, output,
                  ArrayRef<OpFoldResult>{
                      zero,
                      arith::AddIOp::create(b, loc, py, at[0]).getResult(), px,
                      oc},
                  SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
                  SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
              Value first =
                  arith::MulIOp::create(b, loc, at[0], index(b, pooled * 4));
              Value rows = arith::ShLIOp::create(
                  b, loc, extent(b, pooled, width / pool, px), blockLog);
              BankMvoutStridedOp::create(b, loc, i64, memory, converted, first,
                                         rows, index(b, cout * 4), 0, 4,
                                         index(b, 16), blockLog);
            });
    } else {
      loops(
          b, {extent(b, pooled, height / pool, py), quarters},
          [&](OpBuilder &b, ValueRange at) {
            Value c = arith::AddIOp::create(
                b, loc, oc,
                arith::MulIOp::create(b, loc, at[1],
                                      index(b, 16 / elementBytes)));
            Value memory = memref::SubViewOp::create(
                b, loc, output,
                ArrayRef<OpFoldResult>{
                    zero, arith::AddIOp::create(b, loc, py, at[0]).getResult(),
                    px, c},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                       b.getIndexAttr(1), b.getIndexAttr(1)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                       b.getIndexAttr(1), b.getIndexAttr(1)});
            Value first = arith::AddIOp::create(
                b, loc,
                arith::MulIOp::create(b, loc, at[0], index(b, pooled * lines)),
                at[1]);
            Value bytes = arith::MulIOp::create(
                b, loc, extent(b, 16 / elementBytes, cout, c),
                index(b, elementBytes));
            BankMvoutStridedOp::create(
                b, loc, i64, memory, converted, first,
                extent(b, pooled, width / pool, px),
                index(b, cout * elementBytes), 0, lines, bytes,
                arith::ConstantIndexOp::create(b, loc, 0));
          });
    }
    BankReleaseOp::create(b, loc, converted);
    return {};
  };

  bool channelOuter = depthwise || cout <= 16;
  // Ordinary convolutions share each packed spatial tile across output
  // blocks. A single K chunk also shares its im2col bank. Multiple chunks
  // reuse the host packs without keeping extra banks live across outputs.
  // Depthwise blocks read different channels and prepare their weights once.
  auto spatialTiles = [&](OpBuilder &b, Value outerChannel, Value outerScales) {
    pebble::repeat(
        b, loc, 0, height, side, false, {},
        [&](OpBuilder &b, Value y, ValueRange) -> SmallVector<Value> {
          pebble::repeat(
              b, loc, 0, width, side, false, {},
              [&](OpBuilder &b, Value x, ValueRange) -> SmallVector<Value> {
                Value sharedInput;
                if (!channelOuter && chunks.size() == 1)
                  sharedInput = inputBank(b, 0, y, x, zero);
                auto emit = [&](OpBuilder &b, Value oc, Value scales) {
                  Value block = outputTile(b, y, x, oc, scales, sharedInput);
                  if (resident)
                    image = image ? BankTransferOp::create(b, loc, i64, block,
                                                           image)
                                        .getResult()
                                  : block;
                };
                if (channelOuter) {
                  emit(b, outerChannel, outerScales);
                } else {
                  pebble::repeat(b, loc, 0, cout, 16, resident, {},
                                 [&](OpBuilder &b, Value oc,
                                     ValueRange) -> SmallVector<Value> {
                                   Value scales = prepareOutput(b, oc);
                                   emit(b, oc, scales);
                                   if (scales)
                                     BankReleaseOp::create(b, loc, scales);
                                   return {};
                                 });
                }
                if (sharedInput)
                  BankReleaseOp::create(b, loc, sharedInput);
                return {};
              });
          return {};
        });
  };
  if (channelOuter) {
    pebble::repeat(
        builder, loc, 0, cout, 16, resident, {},
        [&](OpBuilder &b, Value oc, ValueRange) -> SmallVector<Value> {
          prepareMultipliers(b, oc);
          Value outerScales = prepareOutput(b, oc);
          spatialTiles(b, oc, outerScales);
          if (outerScales)
            BankReleaseOp::create(b, loc, outerScales);
          return {};
        });
  } else {
    prepareMultipliers(builder, zero);
    spatialTiles(builder, zero, {});
  }
  if (!resident)
    tensorOutput = bufferization::ToTensorOp::create(builder, loc, outputType,
                                                     output, true, true);
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
        op.getKind() == "conv2d_i8_acc" ||
        op.getKind() == "depthwise_conv2d_i8_acc" ||
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
