//===- LowerSpatialOps.cpp - Convolution and pooling on resident banks ----===//
#include "Buckyball/BuckyballOps.h"
#include "Custom/CustomOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using Computation = ::buddy::custom::CustomOp;

namespace {
// Each spatial panel holds sixteen channels per pixel, with at most 64 pixels.
// MAXPOOL can address every panel origin with its six-bit bank row fields.
struct ImageBanks {
  SmallVector<Value> banks;
  int64_t height, width, channels, side;
};

BankCustomOp
compute(OpBuilder &b, Location loc, StringRef kind, ValueRange sources,
        Value destination, bool read,
        std::initializer_list<std::pair<StringRef, int64_t>> attrs) {
  auto op = BankCustomOp::create(b, loc, b.getI64Type(), sources, destination,
                                 ValueRange{}, kind, read);
  for (auto [name, value] : attrs)
    op->setAttr(name, b.getI64IntegerAttr(value));
  return op;
}

LogicalResult convolution(Computation op, const ImageBanks *inputImage,
                          Value nextMultiplier, int64_t pool, bool relu,
                          ImageBanks &outputImage, Value &tensorOutput) {
  bool depthwise = op.getKind() == "depthwise_conv2d_i8";
  auto args = op.getInputs();
  auto inputType = cast<RankedTensorType>(args[0].getType());
  auto weightType = cast<RankedTensorType>(args[1].getType());
  auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
  auto strides = cast<DenseI64ArrayAttr>(op->getAttr("strides")).asArrayRef();
  auto dilation =
      cast<DenseI64ArrayAttr>(op->getAttr("dilations")).asArrayRef();
  auto padding = cast<DenseI64ArrayAttr>(op->getAttr("padding")).asArrayRef();
  if (args.size() != 6 || !inputType.hasStaticShape() ||
      inputType.getRank() != 4 || !weightType.hasStaticShape() ||
      weightType.getRank() != 4 || inputType.getDimSize(0) != 1 ||
      !inputType.getElementType().isF32() ||
      !weightType.getElementType().isInteger(8) ||
      !cast<RankedTensorType>(args[2].getType())
           .getElementType()
           .isInteger(32) ||
      strides[0] != strides[1] || dilation != ArrayRef<int64_t>({1, 1}))
    return op.emitOpError("requires static NHWC/FCHW convolution with INT32 "
                          "bias and unit dilation");
  int64_t cin = inputType.getDimSize(3), cout = weightType.getDimSize(0);
  int64_t kernel = weightType.getDimSize(2), stride = strides[0];
  int64_t height = outputType.getDimSize(1) / pool;
  int64_t width = outputType.getDimSize(2) / pool;
  if (kernel != weightType.getDimSize(3) ||
      (depthwise ? (weightType.getDimSize(1) != 1 || cout != cin)
                 : cin != weightType.getDimSize(1)) ||
      outputType.getDimSize(1) % pool || outputType.getDimSize(2) % pool ||
      kernel > buckyball_target::getBuckyballBallParam("IM2COL", "maxKSize"))
    return op.emitOpError("has incompatible convolution or pooling geometry");
  int64_t tile = 4 / pool;
  while (tile > 0 && (height % tile || width % tile ||
                      ((tile * pool - 1) * stride + kernel) > 8))
    --tile;
  if (!tile)
    return op.emitOpError("convolution window exceeds addressable bank rows");
  int64_t convSide = tile * pool;
  int64_t inputSide = (convSide - 1) * stride + kernel;
  int64_t arrayRows =
      buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
  int64_t rows = (convSide * convSide + arrayRows - 1) / arrayRows * arrayRows;
  int64_t reduction = (kernel * kernel + 15) / 16 * 16;
  int64_t side = std::min<int64_t>(8, std::min(height, width));
  while (side % tile)
    --side;
  int64_t panelsY = (height + side - 1) / side;
  int64_t panelsX = (width + side - 1) / side;
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t outputBanks =
      nextMultiplier ? panelsY * panelsX * ((cout + 15) / 16) : 0;
  int64_t inputBanks = inputImage ? inputImage->banks.size() : 0;
  if (target.bankWidthBits != 128 ||
      inputSide * inputSide * 4 > target.bankDepth ||
      rows * 4 > target.bankDepth || reduction > target.bankDepth ||
      inputBanks + outputBanks + 7 > target.bankNum)
    return op.emitOpError("spatial chain exceeds the configured bank capacity");

  OpBuilder b(op);
  Location loc = op.getLoc();
  Type i64 = b.getI64Type();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(0), one = index(1);
  Value fzero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
  Value izero = arith::ConstantIntOp::create(b, loc, 0, 8);
  Value input;
  if (!inputImage)
    input = pebble::tensorBuffer(b, loc, args[0]);
  Value weight = pebble::tensorBuffer(b, loc, args[1]);
  Value bias = pebble::tensorBuffer(b, loc, args[2]);
  Value weightStep = pebble::tensorBuffer(b, loc, args[5]);
  Value step = tensor::ExtractOp::create(b, loc, args[4],
                                         ValueRange{zero, zero, zero, zero});
  Value multiplier;
  if (nextMultiplier) {
    SmallVector<Value> indices(
        cast<RankedTensorType>(nextMultiplier.getType()).getRank(), zero);
    multiplier = tensor::ExtractOp::create(b, loc, nextMultiplier, indices);
  }
  outputImage = {{}, height, width, cout, side};
  for (int64_t i = 0; i < outputBanks; ++i)
    outputImage.banks.push_back(BankAllocOp::create(b, loc, i64, 1, 1));
  Value output;
  if (!nextMultiplier)
    output = memref::AllocOp::create(
        b, loc, MemRefType::get(outputType.getShape(), b.getF32Type()));
  Value zeroPanel = memref::AllocOp::create(
      b, loc, MemRefType::get({inputSide * inputSide, 16}, b.getI8Type()));
  linalg::FillOp::create(b, loc, izero, zeroPanel);
  Value floatPanel;
  if (!inputImage)
    floatPanel = memref::AllocOp::create(
        b, loc, MemRefType::get({inputSide * inputSide, 16}, b.getF32Type()));
  Value outputPack;
  if (!nextMultiplier)
    outputPack = memref::AllocOp::create(
        b, loc, MemRefType::get({rows, 16}, b.getF32Type()));

  for (int64_t oc = 0; oc < cout; oc += 16) {
    int64_t terms = depthwise ? std::min<int64_t>(16, cout - oc) : cin;
    Value weightPack = memref::AllocOp::create(
        b, loc, MemRefType::get({terms, reduction, 16}, b.getI8Type()));
    linalg::FillOp::create(b, loc, izero, weightPack);
    Value biasPack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getI32Type()));
    Value scalePack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getF32Type()));
    linalg::FillOp::create(
        b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 0, 32)},
        biasPack);
    linalg::FillOp::create(
        b, loc,
        ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1))},
        scalePack);
    for (int64_t lane = 0; lane < std::min<int64_t>(16, cout - oc); ++lane) {
      Value channel = index(oc + lane), laneIndex = index(lane);
      Value biasValue = memref::LoadOp::create(b, loc, bias, channel);
      memref::StoreOp::create(b, loc, biasValue, biasPack, laneIndex);
      Value weightScale = memref::LoadOp::create(
          b, loc, weightStep, ValueRange{channel, zero, zero, zero});
      Value scale = arith::MulFOp::create(b, loc, step, weightScale);
      if (nextMultiplier)
        scale = arith::MulFOp::create(b, loc, scale, multiplier);
      memref::StoreOp::create(b, loc, scale, scalePack, laneIndex);
      auto cLoop = scf::ForOp::create(b, loc, index(depthwise ? lane : 0),
                                      index(depthwise ? lane + 1 : cin), one);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(cLoop.getBody());
      Value c = cLoop.getInductionVar();
      auto kLoop =
          scf::ForOp::create(b, loc, zero, index(kernel * kernel), one);
      b.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      Value ky = arith::DivUIOp::create(b, loc, k, index(kernel));
      Value kx = arith::RemUIOp::create(b, loc, k, index(kernel));
      Value w = memref::LoadOp::create(
          b, loc, weight, ValueRange{channel, depthwise ? zero : c, ky, kx});
      memref::StoreOp::create(b, loc, w, weightPack,
                              ValueRange{c, k, laneIndex});
    }
    Value scales = BankAllocOp::create(b, loc, i64, 1, 1);
    scales = BankMvinOp::create(b, loc, i64, scalePack, scales, 4, 1);
    for (int64_t y = 0; y < height; y += tile) {
      for (int64_t x = 0; x < width; x += tile) {
        Value acc = BankAllocOp::create(b, loc, i64, 1, 1);
        acc = BankMvinOp::create(b, loc, i64, biasPack, acc, 4, 1);
        Value source;
        for (int64_t term = 0; term < terms; ++term) {
          int64_t ic = depthwise ? oc + term : term;
          if (ic % 16 == 0) {
            source = BankAllocOp::create(b, loc, i64, 1, 1);
            if (inputImage) {
              source = BankMvinOp::create(b, loc, i64, zeroPanel, source,
                                          inputSide * inputSide, 1);
              int64_t inPanelsY = (inputImage->height + inputImage->side - 1) /
                                  inputImage->side;
              int64_t inPanelsX =
                  (inputImage->width + inputImage->side - 1) / inputImage->side;
              for (int64_t iy = 0; iy < inputSide; ++iy)
                for (int64_t ix = 0; ix < inputSide; ++ix) {
                  int64_t gy = y * pool * stride + iy - padding[0],
                          gx = x * pool * stride + ix - padding[2];
                  if (gy < 0 || gx < 0 || gy >= inputImage->height ||
                      gx >= inputImage->width)
                    continue;
                  int64_t panel =
                      (ic / 16 * inPanelsY + gy / inputImage->side) *
                          inPanelsX +
                      gx / inputImage->side;
                  int64_t base = gy % inputImage->side * inputImage->side +
                                 gx % inputImage->side;
                  source = compute(b, loc, "maxpool_i8",
                                   ValueRange{inputImage->banks[panel]}, source,
                                   true,
                                   {{"input_side", 1},
                                    {"output_side", 1},
                                    {"kernel", 1},
                                    {"stride", 1},
                                    {"padding", 0},
                                    {"op1_base", base},
                                    {"wr_base", iy * inputSide + ix},
                                    {"output_stride", 1},
                                    {"start_row", 0},
                                    {"start_col", 0}})
                               .getDestinationOut();
                }
            } else {
              linalg::FillOp::create(b, loc, fzero, floatPanel);
              auto pixels = scf::ForOp::create(
                  b, loc, zero, index(inputSide * inputSide), one);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(pixels.getBody());
                Value pixel = pixels.getInductionVar();
                Value iy = arith::AddIOp::create(
                    b, loc,
                    arith::DivUIOp::create(b, loc, pixel, index(inputSide)),
                    index(y * pool * stride - padding[0]));
                Value ix = arith::AddIOp::create(
                    b, loc,
                    arith::RemUIOp::create(b, loc, pixel, index(inputSide)),
                    index(x * pool * stride - padding[2]));
                auto channels = scf::ForOp::create(
                    b, loc, zero, index(std::min<int64_t>(16, cin - ic)), one);
                b.setInsertionPointToStart(channels.getBody());
                Value lane = channels.getInductionVar();
                Value channel = arith::AddIOp::create(b, loc, lane, index(ic));
                Value inBounds = arith::AndIOp::create(
                    b, loc,
                    arith::AndIOp::create(
                        b, loc,
                        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                              iy, zero),
                        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                              iy,
                                              index(inputType.getDimSize(1)))),
                    arith::AndIOp::create(
                        b, loc,
                        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                              ix, zero),
                        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                              ix,
                                              index(inputType.getDimSize(2)))));
                auto valid = scf::IfOp::create(b, loc, inBounds, false);
                b.setInsertionPointToStart(&valid.getThenRegion().front());
                Value value = memref::LoadOp::create(
                    b, loc, input, ValueRange{zero, iy, ix, channel});
                memref::StoreOp::create(b, loc, value, floatPanel,
                                        ValueRange{pixel, lane});
              }
              Value fp = BankAllocOp::create(b, loc, i64, 1, 1);
              fp = BankMvinOp::create(b, loc, i64, floatPanel, fp,
                                      inputSide * inputSide * 4, 1);
              Value inputMultiplier = tensor::ExtractOp::create(
                  b, loc, args[3], ValueRange{zero, zero, zero, zero});
              auto q = BankCustomOp::create(b, loc, i64, ValueRange{fp}, source,
                                            ValueRange{inputMultiplier},
                                            "fp32_to_int8");
              q->setAttr("rows",
                         b.getI64IntegerAttr(inputSide * inputSide * 4));
              source = q.getDestinationOut();
              BankReleaseOp::create(b, loc, fp);
            }
          }
          Value patch = BankAllocOp::create(b, loc, i64, 1, 1);
          patch = compute(b, loc, "im2col_i8", ValueRange{source}, patch, false,
                          {{"input_side", inputSide},
                           {"kernel", kernel},
                           {"stride", stride},
                           {"padding", 0},
                           {"start_row", 0},
                           {"start_col", 0},
                           {"op1_base", 0},
                           {"lane", ic % 16},
                           {"window_start", 0},
                           {"window_count", convSide * convSide}})
                      .getDestinationOut();
          Value packed = BankAllocOp::create(b, loc, i64, 1, 1);
          // Im2col packs M16/K16, while this array consumes tileRows/K16.
          for (int64_t m = 0; m < rows; m += arrayRows)
            for (int64_t k = 0; k < reduction / 16; ++k)
              for (int64_t r = 0; r < arrayRows; r += 4)
                packed =
                    compute(b, loc, "maxpool_i8", ValueRange{patch}, packed,
                            true,
                            {{"input_side", 2},
                             {"output_side", 2},
                             {"kernel", 1},
                             {"stride", 1},
                             {"padding", 0},
                             {"op1_base", k * 16 + m + r},
                             {"wr_base", (m / arrayRows * (reduction / 16) +
                                          k) * arrayRows +
                                             r},
                             {"output_stride", 2},
                             {"start_row", 0},
                             {"start_col", 0}})
                        .getDestinationOut();
          BankReleaseOp::create(b, loc, patch);
          Value w = BankAllocOp::create(b, loc, i64, 1, 1);
          Value slice = memref::SubViewOp::create(
              b, loc, weightPack, ArrayRef<int64_t>{term, 0, 0},
              ArrayRef<int64_t>{1, reduction, 16}, ArrayRef<int64_t>{1, 1, 1});
          w = BankMvinOp::create(b, loc, i64, slice, w, reduction, 1);
          auto matmul =
              compute(b, loc, "matmul_i8_block", ValueRange{packed, w}, acc,
                      true, {{"rows", rows}, {"k", reduction}});
          matmul->setAttr("first", b.getBoolAttr(term == 0));
          matmul->setAttr("last", b.getBoolAttr(term + 1 == terms));
          acc = matmul.getDestinationOut();
          BankReleaseOp::create(b, loc, packed);
          BankReleaseOp::create(b, loc, w);
          if (ic % 16 == 15 || term + 1 == terms)
            BankReleaseOp::create(b, loc, source);
        }
        Value restored = BankAllocOp::create(b, loc, i64, 1, 1);
        if (nextMultiplier) {
          auto q = compute(b, loc, "int32_to_int8", ValueRange{acc, scales},
                           restored, false, {{"rows", rows}, {"wr_base", 0}});
          q->setAttr("relu", b.getBoolAttr(relu));
          restored = q.getDestinationOut();
          int64_t panel = (oc / 16 * panelsY + y / side) * panelsX + x / side;
          outputImage.banks[panel] =
              compute(b, loc, "maxpool_i8", ValueRange{restored},
                      outputImage.banks[panel], true,
                      {{"input_side", convSide},
                       {"output_side", tile},
                       {"kernel", pool},
                       {"stride", pool},
                       {"padding", 0},
                       {"op1_base", 0},
                       {"wr_base", y % side * side + x % side},
                       {"output_stride", side},
                       {"start_row", 0},
                       {"start_col", 0}})
                  .getDestinationOut();
        } else {
          restored = compute(b, loc, "int32_to_fp32", ValueRange{acc, scales},
                             restored, false, {{"rows", rows}})
                         .getDestinationOut();
          restored = BankMvoutOp::create(b, loc, i64, outputPack, restored,
                                         rows * 4, 1);
          for (int64_t iy = 0; iy < tile; ++iy)
            for (int64_t ix = 0; ix < tile; ++ix) {
              auto channels = scf::ForOp::create(
                  b, loc, zero, index(std::min<int64_t>(16, cout - oc)), one);
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(channels.getBody());
              Value c = channels.getInductionVar();
              Value value = memref::LoadOp::create(
                  b, loc, outputPack, ValueRange{index(iy * tile + ix), c});
              memref::StoreOp::create(
                  b, loc, value, output,
                  ValueRange{zero, index(y + iy), index(x + ix),
                             arith::AddIOp::create(b, loc, c, index(oc))});
            }
        }
        BankReleaseOp::create(b, loc, restored);
        BankReleaseOp::create(b, loc, acc);
      }
    }
    BankReleaseOp::create(b, loc, scales);
  }
  if (inputImage)
    for (Value bank : inputImage->banks)
      BankReleaseOp::create(b, loc, bank);
  if (!nextMultiplier)
    tensorOutput = bufferization::ToTensorOp::create(b, loc, outputType, output,
                                                     true, true);
  return success();
}
} // namespace

LogicalResult pebble::lowerSpatialOps(func::FuncOp function,
                                      DenseMap<Value, Value> &residentVectors,
                                      SmallVector<Operation *> &consumed) {
  SmallVector<Computation> convolutions;
  function.walk([&](Computation op) {
    if (op.getKind() == "conv2d_i8" || op.getKind() == "depthwise_conv2d_i8")
      convolutions.push_back(op);
  });
  DenseMap<Value, ImageBanks> images;
  for (auto conv : convolutions) {
    SmallVector<Operation *> links{conv};
    auto group = conv->getAttrOfType<IntegerAttr>("custom.spatial_chain");
    if (group)
      for (Operation *next = conv->getNextNode(); next;
           next = next->getNextNode())
        if (next->getAttr("custom.spatial_chain") == group)
          links.push_back(next);
    Value output = links.back()->getResult(0);
    Value multiplier;
    int64_t pool = 1;
    bool relu = false, flatten = false;
    bool terminalQuant =
        links.back()->getName().getStringRef() == "frontend.fp32_to_int8";
    if (group) {
      if (terminalQuant)
        multiplier = links.back()->getOperand(1);
      else
        multiplier =
            cast<Computation>(*output.getUsers().begin()).getInputs()[3];
      for (Operation *link : links) {
        if (auto op = dyn_cast<Computation>(link)) {
          relu |= op.getKind() == "relu";
          flatten |= op.getKind() == "flatten_nchw";
          if (op.getKind() == "maxpool2d") {
            auto kernel =
                cast<DenseI64ArrayAttr>(op->getAttr("kernel")).asArrayRef();
            auto strides =
                cast<DenseI64ArrayAttr>(op->getAttr("strides")).asArrayRef();
            if (kernel[0] != kernel[1] || strides != kernel)
              return op.emitOpError("requires square non-overlapping pooling");
            pool = kernel[0];
          }
        }
      }
    }
    auto input = images.find(conv.getInputs()[0]);
    ImageBanks result;
    Value tensor;
    if (failed(convolution(conv,
                           input == images.end() ? nullptr : &input->second,
                           multiplier, pool, relu, result, tensor)))
      return failure();
    if (!group) {
      conv.getResult(0).replaceAllUsesWith(tensor);
      consumed.push_back(conv);
      continue;
    }
    OpBuilder b(conv);
    Value bank = result.banks.front();
    if (terminalQuant) {
      Location loc = conv.getLoc();
      auto type = cast<RankedTensorType>(output.getType());
      Value memory = memref::AllocOp::create(
          b, loc, MemRefType::get(type.getShape(), b.getI8Type()));
      Value pack = memref::AllocOp::create(
          b, loc,
          MemRefType::get({result.side * result.side, 16}, b.getI8Type()));
      int64_t py = (result.height + result.side - 1) / result.side;
      int64_t px = (result.width + result.side - 1) / result.side;
      auto index = [&](int64_t n) -> Value {
        return arith::ConstantIndexOp::create(b, loc, n);
      };
      for (int64_t c = 0; c < result.channels; c += 16)
        for (int64_t y = 0; y < result.height; y += result.side)
          for (int64_t x = 0; x < result.width; x += result.side) {
            Value source = result.banks[(c / 16 * py + y / result.side) * px +
                                        x / result.side];
            BankMvoutOp::create(b, loc, b.getI64Type(), pack, source,
                                result.side * result.side, 1);
            for (int64_t iy = 0; iy < std::min(result.side, result.height - y);
                 ++iy)
              for (int64_t ix = 0; ix < std::min(result.side, result.width - x);
                   ++ix) {
                auto channels = scf::ForOp::create(
                    b, loc, index(0),
                    index(std::min<int64_t>(16, result.channels - c)),
                    index(1));
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(channels.getBody());
                Value lane = channels.getInductionVar();
                Value value = memref::LoadOp::create(
                    b, loc, pack,
                    ValueRange{index(iy * result.side + ix), lane});
                memref::StoreOp::create(
                    b, loc, value, memory,
                    ValueRange{index(0), index(y + iy), index(x + ix),
                               arith::AddIOp::create(b, loc, lane, index(c))});
              }
          }
      tensor =
          bufferization::ToTensorOp::create(b, loc, type, memory, true, true);
      output.replaceAllUsesWith(tensor);
      for (Value source : result.banks)
        BankReleaseOp::create(b, loc, source);
    } else if (flatten) {
      if (result.banks.size() != 1 || result.height * result.width != 16 ||
          result.channels != 16 || result.side != result.width)
        return conv.emitOpError(
            "resident NCHW flatten requires a 16x16 INT8 panel");
      Value transposed =
          BankAllocOp::create(b, conv.getLoc(), b.getI64Type(), 1, 1);
      transposed = compute(b, conv.getLoc(), "transpose_i8", ValueRange{bank},
                           transposed, false, {{"rows", 16}, {"cols", 16}})
                       .getDestinationOut();
      BankReleaseOp::create(b, conv.getLoc(), bank);
      bank = transposed;
      residentVectors[output] = bank;
    } else {
      images[output] = result;
    }
    for (Operation *link : links) {
      if (link->getName().getStringRef() == "buddy_trace.end") {
        b.setInsertionPoint(link);
        Operation *trace = b.clone(*link);
        trace->setOperand(0, bank);
        trace->getResult(0).setType(bank.getType());
        trace->removeAttr("custom.spatial_chain");
      }
      consumed.push_back(link);
    }
  }
  return success();
}
