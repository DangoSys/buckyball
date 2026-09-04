#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

#include "llvm/ADT/DenseSet.h"

#include <cmath>
#include <functional>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
constexpr int64_t kTile = 16;

struct Stage {
  Operation *op = nullptr;
  Value input;
  Value rhs;
  Value output;
  Value weight;
  Value bias;
  Value scale;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t inC = 0;
  int64_t outH = 0;
  int64_t outW = 0;
  int64_t outC = 0;
  int64_t kernel = 0;
  int64_t stride = 0;
  int64_t padding = 0;
  int64_t activation = 0;
  float lhsScale = 0;
  float rhsScale = 0;
  float outputScale = 0;
  int64_t addCommon = -1;
  int64_t addCacheRows = 0;
  bool pool = false;
  bool add = false;
  bool average = false;
  bool projection = false;
};

struct CachedPoint {
  int64_t stage;
  Value y;
  Value x;
  Value bank;
  int64_t base;
};

class PointConvRegionPattern : public OpRewritePattern<MegaKernelOp> {
public:
  PointConvRegionPattern(MLIRContext *context)
      : OpRewritePattern<MegaKernelOp>(context, 8) {}

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");
    Block &body = kernel.getBody().front();
    if (body.without_terminator().empty() || !isa<MegaConv2dOp>(body.front()) ||
        !isa<MegaGlobalAvgPoolOp>(*std::prev(body.without_terminator().end())))
      return failure();

    SmallVector<Stage> stages;
    DenseMap<Value, int64_t> producer;
    for (Operation &operation : body.without_terminator()) {
      Stage s;
      s.op = &operation;
      if (auto conv = dyn_cast<MegaConv2dOp>(operation)) {
        auto input = dyn_cast<MemRefType>(conv.getInput().getType());
        auto weight = dyn_cast<MemRefType>(conv.getWeight().getType());
        auto bias = dyn_cast<MemRefType>(conv.getBias().getType());
        auto scale = dyn_cast<MemRefType>(conv.getScale().getType());
        auto output = dyn_cast<MemRefType>(conv.getOutput().getType());
        if (!input || !weight || !bias || !scale || !output ||
            !input.hasStaticShape() || !weight.hasStaticShape() ||
            !bias.hasStaticShape() || !scale.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            weight.getRank() != 4 || output.getRank() != 4 ||
            input.getShape()[0] != 1 || output.getShape()[0] != 1 ||
            !input.getElementType().isInteger(8) ||
            !weight.getElementType().isInteger(8) ||
            !bias.getElementType().isInteger(32) ||
            !scale.getElementType().isF32() ||
            !output.getElementType().isInteger(8) || conv.getStride() <= 0 ||
            conv.getPadLow() < 0 || conv.getPadLow() != conv.getPadHigh() ||
            conv.getActivation() < 0 || conv.getActivation() > 1)
          return conv.emitError("unsupported Conv stage in point scheduler");
        auto in = input.getShape();
        auto w = weight.getShape();
        auto out = output.getShape();
        int64_t k = w[0];
        if (k <= 0 || k > 7 || w[1] != k || w[2] != in[3] || w[3] != out[3] ||
            bias.getShape() != ArrayRef<int64_t>({out[3]}) ||
            scale.getShape() != ArrayRef<int64_t>({out[3]}) || out[3] % kTile ||
            (in[1] + 2 * conv.getPadLow() - k) / conv.getStride() + 1 !=
                out[1] ||
            (in[2] + 2 * conv.getPadLow() - k) / conv.getStride() + 1 != out[2])
          return conv.emitError("inconsistent Conv shape in point scheduler");
        s.input = conv.getInput();
        s.output = conv.getOutput();
        s.weight = conv.getWeight();
        s.bias = conv.getBias();
        s.scale = conv.getScale();
        s.inH = in[1];
        s.inW = in[2];
        s.inC = in[3];
        s.outH = out[1];
        s.outW = out[2];
        s.outC = out[3];
        s.kernel = k;
        s.stride = conv.getStride();
        s.padding = conv.getPadLow();
        s.activation = conv.getActivation();
        s.outputScale = conv.getOutputScale().convertToFloat();
      } else if (auto pool = dyn_cast<MegaMaxPool2dOp>(operation)) {
        auto input = dyn_cast<MemRefType>(pool.getInput().getType());
        auto output = dyn_cast<MemRefType>(pool.getOutput().getType());
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || pool.getFinalOutput() ||
            input.getShape()[0] != 1 || output.getShape()[0] != 1 ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8))
          return pool.emitError("unsupported MaxPool stage in point scheduler");
        auto in = input.getShape();
        auto out = output.getShape();
        if (pool.getKernel() <= 0 || pool.getKernel() > 7 ||
            pool.getStride() <= 0 || pool.getPadding() < 0 || in[3] != out[3] ||
            out[3] % kTile ||
            (in[1] + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                out[1] ||
            (in[2] + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                out[2])
          return pool.emitError(
              "inconsistent MaxPool shape in point scheduler");
        s.input = pool.getInput();
        s.output = pool.getOutput();
        s.inH = in[1];
        s.inW = in[2];
        s.inC = in[3];
        s.outH = out[1];
        s.outW = out[2];
        s.outC = out[3];
        s.kernel = pool.getKernel();
        s.stride = pool.getStride();
        s.padding = pool.getPadding();
        s.pool = true;
      } else if (auto add = dyn_cast<MegaInt8AddOp>(operation)) {
        auto lhs = dyn_cast<MemRefType>(add.getLhs().getType());
        auto rhs = dyn_cast<MemRefType>(add.getRhs().getType());
        auto output = dyn_cast<MemRefType>(add.getOutput().getType());
        float ls = add.getLhsScale().convertToFloat();
        float rs = add.getRhsScale().convertToFloat();
        float os = add.getOutputScale().convertToFloat();
        if (!lhs || !rhs || !output || !lhs.hasStaticShape() || rhs != lhs ||
            output != lhs || lhs.getRank() != 4 || lhs.getShape()[0] != 1 ||
            lhs.getShape()[3] % kTile || !lhs.getElementType().isInteger(8) ||
            add.getActivation() < 0 || add.getActivation() > 1 ||
            !std::isfinite(ls) || !std::isfinite(rs) || !std::isfinite(os) ||
            ls <= 0 || rs <= 0 || os <= 0)
          return add.emitError("unsupported INT8 Add stage in point scheduler");
        auto shape = lhs.getShape();
        s.input = add.getLhs();
        s.rhs = add.getRhs();
        s.output = add.getOutput();
        s.inH = s.outH = shape[1];
        s.inW = s.outW = shape[2];
        s.inC = s.outC = shape[3];
        s.activation = add.getActivation();
        s.lhsScale = ls;
        s.rhsScale = rs;
        s.outputScale = os;
        s.add = true;
      } else if (auto average = dyn_cast<MegaGlobalAvgPoolOp>(operation)) {
        auto input = dyn_cast<MemRefType>(average.getInput().getType());
        auto output = dyn_cast<MemRefType>(average.getOutput().getType());
        float is = average.getInputScale().convertToFloat();
        float os = average.getOutputScale().convertToFloat();
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            input.getShape()[0] != 1 || input.getShape()[3] % kTile ||
            output.getShape() !=
                ArrayRef<int64_t>({1, 1, 1, input.getShape()[3]}) ||
            input.getShape()[1] != input.getShape()[2] ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) || !std::isfinite(is) ||
            !std::isfinite(os) || is <= 0 || os <= 0)
          return average.emitError(
              "unsupported GlobalAvgPool stage in point scheduler");
        auto shape = input.getShape();
        s.input = average.getInput();
        s.output = average.getOutput();
        s.inH = shape[1];
        s.inW = shape[2];
        s.inC = shape[3];
        s.outH = s.outW = 1;
        s.outC = shape[3];
        s.lhsScale = is;
        s.outputScale = os;
        s.average = true;
      } else {
        return operation.emitError("point scheduler accepts Conv, MaxPool, "
                                   "INT8 Add, and GlobalAvgPool");
      }
      if (s.input != kernel.getInput() && !producer.contains(s.input))
        return operation.emitError("stage input has no earlier producer");
      if (s.rhs && s.rhs != kernel.getInput() && !producer.contains(s.rhs))
        return operation.emitError("stage rhs has no earlier producer");
      producer[s.output] = stages.size();
      stages.push_back(s);
    }
    if (stages.back().output != kernel.getOutput())
      return kernel.emitError("final stage must produce MegaKernel output");

    for (auto [index, stage] : llvm::enumerate(stages)) {
      if (!stage.add)
        continue;
      int64_t lhs = producer.lookup(stage.input);
      int64_t rhs = producer.lookup(stage.rhs);
      auto isPrimaryAncestor = [&](int64_t current, int64_t ancestor) {
        while (true) {
          if (current == ancestor)
            return true;
          Value input = stages[current].input;
          if (input == kernel.getInput() || !producer.contains(input))
            return false;
          current = producer.lookup(input);
        }
      };
      if (isPrimaryAncestor(lhs, rhs)) {
        stage.addCommon = rhs;
      } else {
        Stage &projection = stages[rhs];
        if (projection.pool || projection.add || projection.average ||
            projection.kernel != 1 || projection.padding != 0 ||
            projection.input == kernel.getInput() ||
            !producer.contains(projection.input))
          return stage.op->emitError(
              "INT8 Add rhs must be a direct skip or 1x1 projection");
        stage.addCommon = producer.lookup(projection.input);
        stage.projection = true;
        if (!isPrimaryAncestor(lhs, stage.addCommon))
          return stage.op->emitError(
              "INT8 Add branches have no supported common producer");
      }
      stage.addCacheRows = (stages[stage.addCommon].outC + kTile - 1) / kTile;
      if (stage.addCacheRows <= 0 || stage.addCacheRows > 64)
        return stage.op->emitError("INT8 Add cache exceeds one bank");

      int64_t stride = 1;
      int64_t low = 0;
      int64_t high = 0;
      int64_t current = lhs;
      while (current != stage.addCommon) {
        Stage &path = stages[current];
        if (path.pool || path.add || path.average)
          return stage.op->emitError(
              "INT8 Add convolution branch is not a linear Conv chain");
        low = low * path.stride - path.padding;
        high = high * path.stride - path.padding + path.kernel - 1;
        stride *= path.stride;
        if (path.input == kernel.getInput() || !producer.contains(path.input))
          return stage.op->emitError(
              "INT8 Add common producer is not on the convolution branch");
        current = producer.lookup(path.input);
      }
      int64_t rhsStride = stage.projection ? stages[rhs].stride : 1;
      int64_t rhsOffset = stage.projection ? -stages[rhs].padding : 0;
      if (stride != rhsStride || rhsOffset < low || rhsOffset > high)
        return stage.op->emitError(
            "INT8 Add skip point is outside the convolution receptive field");
    }

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth != 64 ||
        target.bankNum != 24)
      return kernel.emitError(
          "point scheduler requires Pebble 24x64x128 banks");

    Location loc = kernel.getLoc();
    b.setInsertionPoint(kernel);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value zeroI8 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value minI8 = b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                              b.getI8IntegerAttr(-128));
    Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                                b.getI32IntegerAttr(0));
    Value zeroPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
    Value minPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
    b.create<linalg::FillOp>(loc, zeroI8, zeroPack);
    b.create<linalg::FillOp>(loc, minI8, minPack);

    DenseMap<Operation *, Value> weights;
    DenseMap<Operation *, Value> biases;
    DenseMap<Operation *, Value> scales;
    SmallVector<Value> hostPacks{zeroPack, minPack};
    for (Stage &s : stages) {
      if (s.pool || s.add || s.average)
        continue;
      int64_t outPanels = s.outC / kTile;
      int64_t paddedCin = (s.inC + kTile - 1) / kTile * kTile;
      int64_t paddedK = (s.kernel * s.kernel + kTile - 1) / kTile * kTile;
      Value wp = b.create<memref::AllocOp>(
          loc, MemRefType::get({outPanels, paddedCin, paddedK, kTile},
                               b.getI8Type()));
      Value bp = b.create<memref::AllocOp>(
          loc, MemRefType::get({outPanels, 4, 4}, b.getI32Type()));
      Value sp = b.create<memref::AllocOp>(
          loc, MemRefType::get({outPanels, 4, 4}, b.getF32Type()));
      hostPacks.append({wp, bp, sp});
      weights[s.op] = wp;
      biases[s.op] = bp;
      scales[s.op] = sp;
      b.create<linalg::FillOp>(loc, zeroI8, wp);
      b.create<linalg::FillOp>(loc, zeroI32, bp);
      auto opLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, outPanels), one);
      b.setInsertionPointToStart(opLoop.getBody());
      Value op = opLoop.getInductionVar();
      auto laneLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
      b.setInsertionPointToStart(laneLoop.getBody());
      Value lane = laneLoop.getInductionVar();
      Value oc = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(loc, op,
                                  b.create<arith::ConstantIndexOp>(loc, kTile)),
          lane);
      Value group = b.create<arith::DivUIOp>(
          loc, lane, b.create<arith::ConstantIndexOp>(loc, 4));
      Value groupLane = b.create<arith::RemUIOp>(
          loc, lane, b.create<arith::ConstantIndexOp>(loc, 4));
      b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, s.bias, oc),
                                bp, ValueRange{op, group, groupLane});
      b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, s.scale, oc),
                                sp, ValueRange{op, group, groupLane});
      b.setInsertionPointAfter(laneLoop);
      auto icLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, s.inC), one);
      b.setInsertionPointToStart(icLoop.getBody());
      Value ic = icLoop.getInductionVar();
      auto kyLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, s.kernel), one);
      b.setInsertionPointToStart(kyLoop.getBody());
      Value ky = kyLoop.getInductionVar();
      auto kxLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, s.kernel), one);
      b.setInsertionPointToStart(kxLoop.getBody());
      Value kx = kxLoop.getInductionVar();
      auto wlLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
      b.setInsertionPointToStart(wlLoop.getBody());
      Value wl = wlLoop.getInductionVar();
      Value sourceOc = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(loc, op,
                                  b.create<arith::ConstantIndexOp>(loc, kTile)),
          wl);
      Value kernelRow = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(
              loc, ky, b.create<arith::ConstantIndexOp>(loc, s.kernel)),
          kx);
      Value w = b.create<memref::LoadOp>(loc, s.weight,
                                         ValueRange{ky, kx, ic, sourceOc});
      b.create<memref::StoreOp>(loc, w, wp, ValueRange{op, ic, kernelRow, wl});
      b.setInsertionPointAfter(opLoop);
    }

    b.setInsertionPoint(kernel);
    SmallVector<int64_t> windowSlot(stages.size(), -1);
    SmallVector<int64_t> rhsWindowSlot(stages.size(), -1);
    SmallVector<int64_t> addCacheSlot(stages.size(), -1);
    int64_t windowRows = 0;
    for (auto [index, s] : llvm::enumerate(stages)) {
      int64_t rows = 0;
      if (!s.pool && !s.average && !s.add && s.input != kernel.getInput())
        rows = s.kernel * s.kernel;
      else if (s.add)
        rows = 2 + s.addCacheRows;
      if (!rows)
        continue;
      if (windowRows % target.bankDepth + rows > target.bankDepth)
        windowRows = (windowRows + target.bankDepth - 1) / target.bankDepth *
                     target.bankDepth;
      windowSlot[index] = windowRows++;
      if (s.add) {
        rhsWindowSlot[index] = windowRows++;
        addCacheSlot[index] = windowRows;
        windowRows += s.addCacheRows;
      } else {
        windowRows += rows - 1;
      }
    }
    SmallVector<Value> windowBanks;
    for (int64_t i = 0;
         i < (windowRows + target.bankDepth - 1) / target.bankDepth; ++i) {
      Value bank = allocBank(b, loc, 1, 1);
      windowBanks.push_back(mvinBank(b, loc, zeroPack, bank, target.bankDepth));
    }

    SmallVector<int64_t> partialSlot(stages.size(), -1);
    int64_t partialRows = 0;
    for (auto [index, s] : llvm::enumerate(stages)) {
      if (s.pool || s.average || s.add)
        continue;
      partialSlot[index] = partialRows;
      partialRows += 4;
    }
    SmallVector<Value> partialBanks;
    for (int64_t i = 0;
         i < (partialRows + target.bankDepth - 1) / target.bankDepth; ++i)
      partialBanks.push_back(allocBank(b, loc, 1, 1));

    Value poolBank = allocBank(b, loc, 1, 1);
    poolBank = mvinBank(b, loc, minPack, poolBank, target.bankDepth);
    Value averageBank = allocBank(b, loc, 1, 1);
    averageBank = mvinBank(b, loc, zeroPack, averageBank, target.bankDepth);
    Value zeroBank = allocBank(b, loc, 1, 1);
    zeroBank = mvinBank(b, loc, zeroPack, zeroBank, target.bankDepth);
    Value addLhsBank = allocBank(b, loc, 1, 1);
    Value addRhsBank = allocBank(b, loc, 1, 1);
    Value addResultBank = allocBank(b, loc, 1, 1);
    Value materializeBank = allocBank(b, loc, 1, 1);
    Value materializeLoadBank = allocBank(b, loc, 1, 1);
    Value materializeStorePack = b.create<memref::AllocOp>(
        loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
    Value materializeLoadPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({1, kTile}, b.getI8Type()));
    DenseSet<int64_t> materialized;

    std::function<LogicalResult(int64_t, Value, Value, Value, Value, Value,
                                CachedPoint *, CachedPoint *)>
        emitPoint;
    emitPoint = [&](int64_t index, Value y, Value x, Value panel,
                    Value destination, Value destinationBase,
                    CachedPoint *cache, CachedPoint *capture) -> LogicalResult {
      Stage &s = stages[index];
      if (materialized.contains(index)) {
        auto laneLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
        b.setInsertionPointToStart(laneLoop.getBody());
        Value lane = laneLoop.getInductionVar();
        Value channel = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, panel, b.create<arith::ConstantIndexOp>(loc, kTile)),
            lane);
        Value value = b.create<memref::LoadOp>(loc, s.output,
                                               ValueRange{zero, y, x, channel});
        b.create<memref::StoreOp>(loc, value, materializeLoadPack,
                                  ValueRange{zero, lane});
        b.setInsertionPointAfter(laneLoop);
        Value loaded =
            mvinBank(b, loc, materializeLoadPack, materializeLoadBank, 1);
        b.create<BankMaxPoolOp>(loc, destination.getType(), loaded, destination,
                                createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(0),
                                createI64Const(b, loc, 0), destinationBase,
                                createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        return success();
      }
      if (cache && cache->stage == index) {
        Value cacheBase = b.create<arith::AddIOp>(
            loc, createI64Const(b, loc, cache->base),
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), panel));
        b.create<BankMaxPoolOp>(loc, destination.getType(), cache->bank,
                                destination, createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(0), cacheBase,
                                destinationBase, createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        return success();
      }
      auto captureResult = [&]() {
        if (!capture || capture->stage != index)
          return;
        Value samePoint = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, y,
                                    capture->y),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, x,
                                    capture->x));
        auto match = b.create<scf::IfOp>(loc, samePoint, false);
        b.setInsertionPointToStart(&match.getThenRegion().front());
        Value cacheBase = b.create<arith::AddIOp>(
            loc, createI64Const(b, loc, capture->base),
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), panel));
        b.create<BankMaxPoolOp>(loc, capture->bank.getType(), destination,
                                capture->bank, createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(0), destinationBase,
                                cacheBase, createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        b.setInsertionPointAfter(match);
      };
      if (s.add) {
        int64_t lhsSlot = windowSlot[index];
        int64_t rhsSlot = rhsWindowSlot[index];
        int64_t cacheSlot = addCacheSlot[index];
        Value lhs = windowBanks[lhsSlot / target.bankDepth];
        Value rhs = windowBanks[rhsSlot / target.bankDepth];
        Value cacheBank = windowBanks[cacheSlot / target.bankDepth];
        Value captureY = y;
        Value captureX = x;
        if (s.projection) {
          Stage &projection = stages[producer.lookup(s.rhs)];
          captureY = b.create<arith::SubIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, y,
                  b.create<arith::ConstantIndexOp>(loc, projection.stride)),
              b.create<arith::ConstantIndexOp>(loc, projection.padding));
          captureX = b.create<arith::SubIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, x,
                  b.create<arith::ConstantIndexOp>(loc, projection.stride)),
              b.create<arith::ConstantIndexOp>(loc, projection.padding));
        }
        CachedPoint branchCache{s.addCommon, captureY, captureX, cacheBank,
                                cacheSlot % target.bankDepth};
        bool commonMaterialized = materialized.contains(s.addCommon);
        if (failed(emitPoint(producer.lookup(s.input), y, x, panel, lhs,
                             createI64Const(b, loc, lhsSlot % target.bankDepth),
                             nullptr,
                             commonMaterialized ? nullptr : &branchCache)))
          return failure();
        if (failed(emitPoint(producer.lookup(s.rhs), y, x, panel, rhs,
                             createI64Const(b, loc, rhsSlot % target.bankDepth),
                             commonMaterialized ? nullptr : &branchCache,
                             nullptr)))
          return failure();
        b.create<BankMaxPoolOp>(
            loc, addLhsBank.getType(), lhs, addLhsBank,
            createI64Const(b, loc, 1), b.getI64IntegerAttr(1),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
            createI64Const(b, loc, lhsSlot % target.bankDepth),
            createI64Const(b, loc, 0), createI64Const(b, loc, 1),
            b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        b.create<BankMaxPoolOp>(
            loc, addRhsBank.getType(), rhs, addRhsBank,
            createI64Const(b, loc, 1), b.getI64IntegerAttr(1),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
            createI64Const(b, loc, rhsSlot % target.bankDepth),
            createI64Const(b, loc, 0), createI64Const(b, loc, 1),
            b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        Value lr = b.create<arith::ConstantOp>(
            loc, b.getF32Type(), b.getF32FloatAttr(s.lhsScale / s.outputScale));
        Value rr = b.create<arith::ConstantOp>(
            loc, b.getF32Type(), b.getF32FloatAttr(s.rhsScale / s.outputScale));
        b.create<BankInt8AddOp>(loc, addResultBank.getType(), addLhsBank,
                                addRhsBank, addResultBank,
                                createI64Const(b, loc, 1), lr, rr,
                                b.getBoolAttr(s.activation == 1));
        b.create<BankMaxPoolOp>(loc, destination.getType(), addResultBank,
                                destination, createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(0),
                                createI64Const(b, loc, 0), destinationBase,
                                createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        captureResult();
        return success();
      }

      if (s.pool) {
        Value source = mvinBank(b, loc, minPack, poolBank, target.bankDepth);
        auto kyLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, s.kernel), one);
        b.setInsertionPointToStart(kyLoop.getBody());
        Value ky = kyLoop.getInductionVar();
        auto kxLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, s.kernel), one);
        b.setInsertionPointToStart(kxLoop.getBody());
        Value kx = kxLoop.getInductionVar();
        Value iy = b.create<arith::SubIOp>(
            loc,
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, y, b.create<arith::ConstantIndexOp>(loc, s.stride)),
                ky),
            b.create<arith::ConstantIndexOp>(loc, s.padding));
        Value ix = b.create<arith::SubIOp>(
            loc,
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, x, b.create<arith::ConstantIndexOp>(loc, s.stride)),
                kx),
            b.create<arith::ConstantIndexOp>(loc, s.padding));
        Value yOk = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iy, zero),
            b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, iy,
                b.create<arith::ConstantIndexOp>(loc, s.inH)));
        Value xOk = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ix, zero),
            b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, ix,
                b.create<arith::ConstantIndexOp>(loc, s.inW)));
        auto valid = b.create<scf::IfOp>(
            loc, b.create<arith::AndIOp>(loc, yOk, xOk), false);
        b.setInsertionPointToStart(&valid.getThenRegion().front());
        Value sourceRow = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, ky, b.create<arith::ConstantIndexOp>(loc, s.kernel)),
            kx);
        if (failed(emitPoint(
                producer.lookup(s.input), iy, ix, panel, source,
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), sourceRow),
                cache, capture)))
          return failure();
        b.setInsertionPointAfter(valid);
        b.setInsertionPointAfter(kyLoop);
        b.create<BankMaxPoolOp>(
            loc, destination.getType(), source, destination,
            createI64Const(b, loc, 1), b.getI64IntegerAttr(s.kernel),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(s.kernel),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
            createI64Const(b, loc, 0), destinationBase,
            createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(0));
        captureResult();
        return success();
      }

      if (s.average) {
        int64_t rows = s.inH * s.inW;
        if (rows > target.bankDepth)
          return s.op->emitError("GlobalAvg input panel exceeds one bank");
        Value source =
            mvinBank(b, loc, zeroPack, averageBank, target.bankDepth);
        auto rowLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, rows), one);
        b.setInsertionPointToStart(rowLoop.getBody());
        Value row = rowLoop.getInductionVar();
        Value iy = b.create<arith::DivUIOp>(
            loc, row, b.create<arith::ConstantIndexOp>(loc, s.inW));
        Value ix = b.create<arith::RemUIOp>(
            loc, row, b.create<arith::ConstantIndexOp>(loc, s.inW));
        if (failed(emitPoint(
                producer.lookup(s.input), iy, ix, panel, source,
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), row), cache,
                capture)))
          return failure();
        b.setInsertionPointAfter(rowLoop);
        Value onesPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, kTile}, b.getI8Type()));
        Value biasPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getI32Type()));
        Value scalePack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getF32Type()));
        b.create<linalg::FillOp>(loc, zeroI8, onesPack);
        b.create<linalg::FillOp>(loc, zeroI32, biasPack);
        Value ratio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(s.lhsScale / (rows * s.outputScale)));
        b.create<linalg::FillOp>(loc, ratio, scalePack);
        auto onesLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, rows), one);
        b.setInsertionPointToStart(onesLoop.getBody());
        Value i = onesLoop.getInductionVar();
        b.create<memref::StoreOp>(
            loc,
            b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                        b.getI8IntegerAttr(1)),
            onesPack,
            ValueRange{
                b.create<arith::DivUIOp>(
                    loc, i, b.create<arith::ConstantIndexOp>(loc, kTile)),
                b.create<arith::RemUIOp>(
                    loc, i, b.create<arith::ConstantIndexOp>(loc, kTile))});
        b.setInsertionPointAfter(onesLoop);
        Value ones = allocBank(b, loc, 1, 1);
        ones = mvinBank(b, loc, onesPack, ones, 4);
        Value bias = allocBank(b, loc, 1, 1);
        bias = mvinBank(b, loc, biasPack, bias, 4);
        b.create<BankSMatMulBiasOp>(loc, bias.getType(), bias,
                                    createI64Const(b, loc, 0));
        Value scale = allocBank(b, loc, 1, 1);
        scale = mvinBank(b, loc, scalePack, scale, 4);
        Value result = allocBank(b, loc, 1, 1);
        b.create<BankSMatMulOp>(
            loc, result.getType(), ones, source, result,
            createI64ConstU(b, loc, matrixRs2(1, kTile, target.bankDepth)),
            createI1Const(b, loc, true), createI1Const(b, loc, true),
            createI64Const(b, loc, 0));
        Value quantized = allocBank(b, loc, 1, 1);
        b.create<BankQuantI32ToI8Op>(
            loc, quantized.getType(), result, scale, quantized,
            createI64Const(b, loc, 4), createI64Const(b, loc, 0),
            createI64Const(b, loc, 0), b.getI64IntegerAttr(1),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
            b.getBoolAttr(false));
        b.create<BankMaxPoolOp>(loc, destination.getType(), quantized,
                                destination, createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(0),
                                createI64Const(b, loc, 0), destinationBase,
                                createI64Const(b, loc, 1),
                                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
        releaseBank(b, loc, ones);
        releaseBank(b, loc, bias);
        releaseBank(b, loc, scale);
        releaseBank(b, loc, result);
        releaseBank(b, loc, quantized);
        b.create<memref::DeallocOp>(loc, onesPack);
        b.create<memref::DeallocOp>(loc, biasPack);
        b.create<memref::DeallocOp>(loc, scalePack);
        captureResult();
        return success();
      }

      int64_t paddedCin = (s.inC + kTile - 1) / kTile * kTile;
      int64_t paddedK = (s.kernel * s.kernel + kTile - 1) / kTile * kTile;
      SmallVector<OpFoldResult> parameterOffsets = {panel, b.getIndexAttr(0),
                                                    b.getIndexAttr(0)};
      SmallVector<OpFoldResult> parameterSizes = {
          b.getIndexAttr(1), b.getIndexAttr(4), b.getIndexAttr(4)};
      SmallVector<OpFoldResult> parameterStrides(3, b.getIndexAttr(1));
      int64_t partialOffset = partialSlot[index];
      if (partialOffset < 0)
        return s.op->emitError("Conv stage has no partial-sum allocation");
      Value partialBank = partialBanks[partialOffset / target.bankDepth];
      Value partialBase =
          createI64Const(b, loc, partialOffset % target.bankDepth);

      auto panelLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, paddedCin / kTile),
          one);
      b.setInsertionPointToStart(panelLoop.getBody());
      Value inputPanel = panelLoop.getInductionVar();
      Value source;
      Value inputPack;
      Value inputWindow;
      Value inputWindowBase;
      if (s.input == kernel.getInput()) {
        if (s.inC > kTile)
          return s.op->emitError("MegaKernel input exceeds one channel panel");
        inputPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
        b.create<linalg::FillOp>(loc, zeroI8, inputPack);
      } else {
        int64_t slot = windowSlot[index];
        if (slot < 0)
          return s.op->emitError("Conv stage has no input-window allocation");
        inputWindow = windowBanks[slot / target.bankDepth];
        inputWindowBase = createI64Const(b, loc, slot % target.bankDepth);
        b.create<BankMaxPoolOp>(
            loc, inputWindow.getType(), zeroBank, inputWindow,
            createI64Const(b, loc, s.kernel * s.kernel),
            b.getI64IntegerAttr(s.kernel), b.getI64IntegerAttr(s.kernel),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
            b.getI64IntegerAttr(0), createI64Const(b, loc, 0), inputWindowBase,
            createI64Const(b, loc, s.kernel), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(0));
      }
      auto kyLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, s.kernel), one);
      b.setInsertionPointToStart(kyLoop.getBody());
      Value ky = kyLoop.getInductionVar();
      auto kxLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, s.kernel), one);
      b.setInsertionPointToStart(kxLoop.getBody());
      Value kx = kxLoop.getInductionVar();
      Value iy = b.create<arith::SubIOp>(
          loc,
          b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, y, b.create<arith::ConstantIndexOp>(loc, s.stride)),
              ky),
          b.create<arith::ConstantIndexOp>(loc, s.padding));
      Value ix = b.create<arith::SubIOp>(
          loc,
          b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, x, b.create<arith::ConstantIndexOp>(loc, s.stride)),
              kx),
          b.create<arith::ConstantIndexOp>(loc, s.padding));
      Value yOk = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iy, zero),
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, iy,
              b.create<arith::ConstantIndexOp>(loc, s.inH)));
      Value xOk = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ix, zero),
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, ix,
              b.create<arith::ConstantIndexOp>(loc, s.inW)));
      auto valid = b.create<scf::IfOp>(
          loc, b.create<arith::AndIOp>(loc, yOk, xOk), false);
      b.setInsertionPointToStart(&valid.getThenRegion().front());
      if (s.input == kernel.getInput()) {
        auto laneLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, s.inC), one);
        b.setInsertionPointToStart(laneLoop.getBody());
        Value lane = laneLoop.getInductionVar();
        Value v = b.create<memref::LoadOp>(loc, kernel.getInput(),
                                           ValueRange{zero, iy, ix, lane});
        Value sourceRow = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, ky, b.create<arith::ConstantIndexOp>(loc, s.kernel)),
            kx);
        b.create<memref::StoreOp>(loc, v, inputPack,
                                  ValueRange{sourceRow, lane});
        b.setInsertionPointAfter(laneLoop);
      } else if (failed(emitPoint(
                     producer.lookup(s.input), iy, ix, inputPanel, inputWindow,
                     b.create<arith::AddIOp>(
                         loc, inputWindowBase,
                         b.create<arith::IndexCastOp>(
                             loc, b.getI64Type(),
                             b.create<arith::AddIOp>(
                                 loc,
                                 b.create<arith::MulIOp>(
                                     loc, ky,
                                     b.create<arith::ConstantIndexOp>(
                                         loc, s.kernel)),
                                 kx))),
                     cache, capture))) {
        return failure();
      }
      b.setInsertionPointAfter(valid);
      b.setInsertionPointAfter(kyLoop);
      if (s.input == kernel.getInput()) {
        source = allocBank(b, loc, 1, 1);
        source = mvinBank(b, loc, inputPack, source, target.bankDepth);
        b.create<memref::DeallocOp>(loc, inputPack);
      } else {
        source = allocBank(b, loc, 1, 1);
        source =
            b.create<BankMaxPoolOp>(
                 loc, source.getType(), inputWindow, source,
                 createI64Const(b, loc, s.kernel * s.kernel),
                 b.getI64IntegerAttr(s.kernel), b.getI64IntegerAttr(s.kernel),
                 b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                 b.getI64IntegerAttr(0), inputWindowBase,
                 createI64Const(b, loc, 0), createI64Const(b, loc, s.kernel),
                 b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                .getOutBankOut();
      }

      auto firstPanel = b.create<scf::IfOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, inputPanel,
                                  zero),
          true);
      b.setInsertionPointToStart(&firstPanel.getThenRegion().front());
      Value biasSlice = b.create<memref::SubViewOp>(
          loc, biases.lookup(s.op), parameterOffsets, parameterSizes,
          parameterStrides);
      Value biasPack = b.create<memref::CollapseShapeOp>(
          loc, biasSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});
      Value biasBank = allocBank(b, loc, 1, 1);
      biasBank = mvinBank(b, loc, biasPack, biasBank, 4);
      biasBank = b.create<BankSMatMulBiasOp>(loc, biasBank.getType(), biasBank,
                                             createI64Const(b, loc, 0))
                     .getBiasBankOut();
      releaseBank(b, loc, biasBank);
      b.setInsertionPointToStart(&firstPanel.getElseRegion().front());
      b.create<BankSMatMulBiasOp>(loc, partialBank.getType(), partialBank,
                                  partialBase);
      b.setInsertionPointAfter(firstPanel);

      Value patch = allocBank(b, loc, 1, 1);
      Value weight = allocBank(b, loc, 1, 1);
      auto laneLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
      b.setInsertionPointToStart(laneLoop.getBody());
      Value lane = laneLoop.getInductionVar();
      Value inputChannel = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(loc, inputPanel,
                                  b.create<arith::ConstantIndexOp>(loc, kTile)),
          lane);
      b.create<BankIm2colOp>(
          loc, patch.getType(), source, patch, createI64Const(b, loc, s.kernel),
          createI64Const(b, loc, s.kernel), createI64Const(b, loc, 1),
          createI64Const(b, loc, 0), createI64Const(b, loc, 0),
          b.create<arith::IndexCastOp>(loc, b.getI64Type(), lane),
          b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
          b.getI64IntegerAttr(0), b.getI64IntegerAttr(1));
      SmallVector<OpFoldResult> weightOffsets = {
          panel, inputChannel, b.getIndexAttr(0), b.getIndexAttr(0)};
      SmallVector<OpFoldResult> weightSizes = {
          b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(paddedK),
          b.getIndexAttr(kTile)};
      SmallVector<OpFoldResult> weightStrides(4, b.getIndexAttr(1));
      Value weightSlice = b.create<memref::SubViewOp>(
          loc, weights.lookup(s.op), weightOffsets, weightSizes, weightStrides);
      Value weightPack = b.create<memref::CollapseShapeOp>(
          loc, weightSlice, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
      mvinBank(b, loc, weightPack, weight, paddedK);
      b.create<BankSMatMulOp>(
          loc, partialBank.getType(), patch, weight, partialBank,
          createI64ConstU(b, loc, matrixRs2(1, kTile, paddedK)),
          createI1Const(b, loc, true), createI1Const(b, loc, true),
          partialBase);
      b.setInsertionPointAfter(laneLoop);
      releaseBank(b, loc, patch);
      releaseBank(b, loc, weight);

      Value finalPanel = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, inputPanel,
          b.create<arith::ConstantIndexOp>(loc, paddedCin / kTile - 1));
      auto quantize = b.create<scf::IfOp>(loc, finalPanel, false);
      b.setInsertionPointToStart(&quantize.getThenRegion().front());
      Value scaleSlice = b.create<memref::SubViewOp>(
          loc, scales.lookup(s.op), parameterOffsets, parameterSizes,
          parameterStrides);
      Value scalePack = b.create<memref::CollapseShapeOp>(
          loc, scaleSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});
      Value scaleBank = allocBank(b, loc, 1, 1);
      scaleBank = mvinBank(b, loc, scalePack, scaleBank, 4);
      Value quantized = allocBank(b, loc, 1, 1);
      b.create<BankQuantI32ToI8Op>(
          loc, quantized.getType(), partialBank, scaleBank, quantized,
          createI64Const(b, loc, 4), createI64Const(b, loc, 0), partialBase,
          b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
          b.getI64IntegerAttr(1), b.getBoolAttr(s.activation == 1));
      b.create<BankMaxPoolOp>(loc, destination.getType(), quantized,
                              destination, createI64Const(b, loc, 1),
                              b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                              b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                              b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
                              destinationBase, createI64Const(b, loc, 1),
                              b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
      releaseBank(b, loc, scaleBank);
      releaseBank(b, loc, quantized);
      b.setInsertionPointAfter(quantize);
      releaseBank(b, loc, source);
      b.setInsertionPointAfter(panelLoop);
      captureResult();
      return success();
    };

    auto materializeStage = [&](int64_t index) -> LogicalResult {
      Stage &s = stages[index];
      int64_t panels = s.outC / kTile;
      int64_t points = s.outH * s.outW;
      auto panelLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, panels), one);
      b.setInsertionPointToStart(panelLoop.getBody());
      Value panel = panelLoop.getInductionVar();
      auto batchLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, points),
          b.create<arith::ConstantIndexOp>(loc, target.bankDepth));
      b.setInsertionPointToStart(batchLoop.getBody());
      Value batch = batchLoop.getInductionVar();
      Value batchBank =
          mvinBank(b, loc, zeroPack, materializeBank, target.bankDepth);
      auto rowLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, target.bankDepth),
          one);
      b.setInsertionPointToStart(rowLoop.getBody());
      Value row = rowLoop.getInductionVar();
      Value point = b.create<arith::AddIOp>(loc, batch, row);
      auto valid = b.create<scf::IfOp>(
          loc,
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, point,
              b.create<arith::ConstantIndexOp>(loc, points)),
          false);
      b.setInsertionPointToStart(&valid.getThenRegion().front());
      Value y = b.create<arith::DivUIOp>(
          loc, point, b.create<arith::ConstantIndexOp>(loc, s.outW));
      Value x = b.create<arith::RemUIOp>(
          loc, point, b.create<arith::ConstantIndexOp>(loc, s.outW));
      if (failed(
              emitPoint(index, y, x, panel, batchBank,
                        b.create<arith::IndexCastOp>(loc, b.getI64Type(), row),
                        nullptr, nullptr)))
        return failure();
      b.setInsertionPointAfter(valid);
      b.setInsertionPointAfter(rowLoop);
      mvoutBank(b, loc, materializeStorePack, batchBank, target.bankDepth);
      b.create<FenceOp>(loc);
      auto storeLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, target.bankDepth),
          one);
      b.setInsertionPointToStart(storeLoop.getBody());
      row = storeLoop.getInductionVar();
      point = b.create<arith::AddIOp>(loc, batch, row);
      valid = b.create<scf::IfOp>(
          loc,
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, point,
              b.create<arith::ConstantIndexOp>(loc, points)),
          false);
      b.setInsertionPointToStart(&valid.getThenRegion().front());
      y = b.create<arith::DivUIOp>(
          loc, point, b.create<arith::ConstantIndexOp>(loc, s.outW));
      x = b.create<arith::RemUIOp>(
          loc, point, b.create<arith::ConstantIndexOp>(loc, s.outW));
      auto laneLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
      b.setInsertionPointToStart(laneLoop.getBody());
      Value lane = laneLoop.getInductionVar();
      Value channel = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(loc, panel,
                                  b.create<arith::ConstantIndexOp>(loc, kTile)),
          lane);
      Value value = b.create<memref::LoadOp>(loc, materializeStorePack,
                                             ValueRange{row, lane});
      b.create<memref::StoreOp>(loc, value, s.output,
                                ValueRange{zero, y, x, channel});
      b.setInsertionPointAfter(laneLoop);
      b.setInsertionPointAfter(valid);
      b.setInsertionPointAfter(storeLoop);
      b.setInsertionPointAfter(batchLoop);
      b.setInsertionPointAfter(panelLoop);
      materialized.insert(index);
      return success();
    };

    for (auto [index, s] : llvm::enumerate(stages)) {
      if ((s.pool || s.add) && failed(materializeStage(index)))
        return failure();
    }

    Stage &finalStage = stages.back();
    int64_t panels = finalStage.outC / kTile;
    Value outputBank = allocBank(b, loc, 1, 1);
    outputBank = mvinBank(b, loc, zeroPack, outputBank, target.bankDepth);
    auto panelLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, panels), one);
    b.setInsertionPointToStart(panelLoop.getBody());
    if (failed(emitPoint(stages.size() - 1, zero, zero,
                         panelLoop.getInductionVar(), outputBank,
                         b.create<arith::IndexCastOp>(
                             loc, b.getI64Type(), panelLoop.getInductionVar()),
                         nullptr, nullptr)))
      return failure();
    b.setInsertionPointAfter(panelLoop);
    Value packed = b.create<memref::AllocOp>(
        loc, MemRefType::get({panels, kTile}, b.getI8Type()));
    mvoutBank(b, loc, packed, outputBank, panels);
    b.create<FenceOp>(loc);
    auto rowLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, panels), one);
    b.setInsertionPointToStart(rowLoop.getBody());
    auto laneLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
    b.setInsertionPointToStart(laneLoop.getBody());
    Value channel = b.create<arith::AddIOp>(
        loc,
        b.create<arith::MulIOp>(loc, rowLoop.getInductionVar(),
                                b.create<arith::ConstantIndexOp>(loc, kTile)),
        laneLoop.getInductionVar());
    Value value = b.create<memref::LoadOp>(
        loc, packed,
        ValueRange{rowLoop.getInductionVar(), laneLoop.getInductionVar()});
    b.create<memref::StoreOp>(loc, value, kernel.getOutput(),
                              ValueRange{zero, zero, zero, channel});
    b.setInsertionPointAfter(rowLoop);
    releaseBank(b, loc, outputBank);
    for (Value bank : windowBanks)
      releaseBank(b, loc, bank);
    for (Value bank : partialBanks)
      releaseBank(b, loc, bank);
    releaseBank(b, loc, poolBank);
    releaseBank(b, loc, averageBank);
    releaseBank(b, loc, zeroBank);
    releaseBank(b, loc, addLhsBank);
    releaseBank(b, loc, addRhsBank);
    releaseBank(b, loc, addResultBank);
    releaseBank(b, loc, materializeBank);
    releaseBank(b, loc, materializeLoadBank);
    for (Value pack : hostPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.create<memref::DeallocOp>(loc, materializeStorePack);
    b.create<memref::DeallocOp>(loc, materializeLoadPack);
    b.create<memref::DeallocOp>(loc, packed);
    b.eraseOp(kernel);
    return success();
  }
};
} // namespace

namespace mlir::buddy {
void populatePebblePointConvRegionToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PointConvRegionPattern>(patterns.getContext());
}
} // namespace mlir::buddy
