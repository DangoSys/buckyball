#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

#include <cmath>

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult GemminiMemConvOp::lowerToBankSSA(PatternRewriter &b) {
  GemminiMemConvOp op = *this;
  auto in = cast<MemRefType>(op.getInput().getType());
  auto w = cast<MemRefType>(op.getWeight().getType());
  auto bias = cast<MemRefType>(op.getBias().getType());
  auto out = cast<MemRefType>(op.getOutput().getType());
  bool integer = in.getElementType().isInteger(8) &&
                 bias.getElementType().isInteger(32) &&
                 out.getElementType().isInteger(32);
  bool floating = in.getElementType().isF32() &&
                  bias.getElementType().isF32() && out.getElementType().isF32();
  if (in.getRank() != 4 || w.getRank() != 2 || bias.getRank() != 1 ||
      out.getRank() != 4 || !in.hasStaticShape() || !w.hasStaticShape() ||
      !bias.hasStaticShape() || !out.hasStaticShape() ||
      !w.getElementType().isInteger(8) || (!integer && !floating))
    return op.emitError(
        "requires static NCHW input/output and I8 KxN weights; "
        "use I8 input with I32 output/bias, or F32 input/output/bias");

  int64_t C = in.getShape()[1], H = in.getShape()[2], W = in.getShape()[3];
  int64_t OH = out.getShape()[2], OW = out.getShape()[3];
  int64_t kh = op.getKh(), kw = op.getKw(), stride = op.getStride();
  int64_t pad = op.getPadding();
  int64_t M = in.getShape()[0] * OH * OW, K = C * kh * kw;
  int64_t N = w.getShape()[1], kTiles = (K + 15) / 16;
  if (w.getShape()[0] != K || bias.getShape()[0] != N ||
      out.getShape()[0] != in.getShape()[0] || out.getShape()[1] != N ||
      OH != (H + 2 * pad - kh) / stride + 1 ||
      OW != (W + 2 * pad - kw) / stride + 1)
    return op.emitError("conv shape mismatch");
  float inScale = op.getInputScale().convertToFloat();
  float wScale = op.getWeightScale().convertToFloat();
  if (!std::isfinite(inScale) || inScale <= 0 || !std::isfinite(wScale) ||
      wScale <= 0)
    return op.emitError("scales must be finite and positive");

  if (integer && (inScale != 1.0f || wScale != 1.0f))
    return op.emitError("native INT8 convolution requires unit scales");

  const auto &target = buckyball_target::getBuckyballTarget();
  if (target.bankWidthBits != 128 || target.bankDepth < 16)
    return op.emitError(
        "GemminiBall requires a 128-bit bank with at least 16 lines");

  Location loc = op.getLoc();
  auto idx = [&](int64_t v) -> Value {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };
  using P = arith::CmpIPredicate;
  Value zero = idx(0), one = idx(1), sixteen = idx(16);
  Value zeroI8 = b.create<arith::ConstantIntOp>(loc, 0, 8);
  IntegerAttr align = b.getI64IntegerAttr(64);

  // Quantized im2col panels, one 16x16 tile per (M tile, K tile).
  Value panels = b.create<memref::AllocOp>(
      loc, MemRefType::get({(M + 15) / 16 * kTiles, 16, 16}, b.getI8Type()),
      align);
  b.create<linalg::FillOp>(loc, zeroI8, panels);
  auto rows = b.create<scf::ForOp>(loc, zero, idx(M), one);
  b.setInsertionPointToStart(rows.getBody());
  Value r = rows.getInductionVar();
  Value y0 = b.create<arith::SubIOp>(
      loc,
      b.create<arith::MulIOp>(
          loc,
          b.create<arith::RemUIOp>(
              loc, b.create<arith::DivUIOp>(loc, r, idx(OW)), idx(OH)),
          idx(stride)),
      idx(pad));
  Value x0 = b.create<arith::SubIOp>(
      loc,
      b.create<arith::MulIOp>(loc, b.create<arith::RemUIOp>(loc, r, idx(OW)),
                              idx(stride)),
      idx(pad));
  auto inners = b.create<scf::ForOp>(loc, zero, idx(K), one);
  b.setInsertionPointToStart(inners.getBody());
  Value i = inners.getInductionVar();
  Value y = b.create<arith::AddIOp>(
      loc, y0, b.create<arith::DivUIOp>(loc, i, idx(kw * C)));
  Value x = b.create<arith::AddIOp>(
      loc, x0,
      b.create<arith::RemUIOp>(loc, b.create<arith::DivUIOp>(loc, i, idx(C)),
                               idx(kw)));
  Value yIn = b.create<arith::AndIOp>(
      loc, b.create<arith::CmpIOp>(loc, P::sge, y, zero),
      b.create<arith::CmpIOp>(loc, P::slt, y, idx(H)));
  Value xIn = b.create<arith::AndIOp>(
      loc, b.create<arith::CmpIOp>(loc, P::sge, x, zero),
      b.create<arith::CmpIOp>(loc, P::slt, x, idx(W)));
  auto inside =
      b.create<scf::IfOp>(loc, b.create<arith::AndIOp>(loc, yIn, xIn));
  b.setInsertionPointToStart(&inside.getThenRegion().front());
  Value v = b.create<memref::LoadOp>(
      loc, op.getInput(),
      ValueRange{b.create<arith::DivUIOp>(loc, r, idx(OH * OW)),
                 b.create<arith::RemUIOp>(loc, i, idx(C)), y, x});
  if (!integer) {
    v = b.create<arith::DivFOp>(
        loc, v, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(inScale)));
    v = b.create<arith::MaximumFOp>(
        loc, v, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(-128)));
    v = b.create<arith::MinimumFOp>(
        loc, v, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(127)));
    v = b.create<math::RoundEvenOp>(loc, v);
    v = b.create<arith::FPToSIOp>(loc, b.getI32Type(), v);
    v = b.create<arith::TruncIOp>(loc, b.getI8Type(), v);
  }
  Value panel = b.create<arith::AddIOp>(
      loc,
      b.create<arith::MulIOp>(loc, b.create<arith::DivUIOp>(loc, r, sixteen),
                              idx(kTiles)),
      b.create<arith::DivUIOp>(loc, i, sixteen));
  b.create<memref::StoreOp>(
      loc, v, panels,
      ValueRange{panel, b.create<arith::RemUIOp>(loc, r, sixteen),
                 b.create<arith::RemUIOp>(loc, i, sixteen)});
  b.setInsertionPointAfter(rows);

  auto tileType = MemRefType::get({16, 16}, b.getI8Type());
  Value dTile = b.create<memref::AllocOp>(loc, tileType, align);
  Value wTile = b.create<memref::AllocOp>(loc, tileType, align);
  Value acc = b.create<memref::AllocOp>(
      loc, MemRefType::get({16, 16}, b.getI32Type()), align);
  b.create<linalg::FillOp>(loc, zeroI8, dTile);
  auto meta = b.create<memref::ExtractStridedMetadataOp>(loc, op.getWeight());
  Value wStride = meta.getStrides()[0];
  Value wDirect = b.create<arith::AndIOp>(
      loc,
      b.create<arith::CmpIOp>(
          loc, P::eq, b.create<arith::RemUIOp>(loc, wStride, sixteen), zero),
      b.create<arith::CmpIOp>(loc, P::eq, meta.getStrides()[1], one));
  Value wLines = b.create<arith::IndexCastOp>(
      loc, b.getI64Type(), b.create<arith::DivUIOp>(loc, wStride, sixteen));
  Value iter = createI64Const(b, loc, 16), base = createI64Const(b, loc, 0);
  // Weight-stationary dataflow.
  b.create<GemminiInstructionOp>(loc, b.getStringAttr("GEMMINI_CONFIG"), base,
                                 createI64Const(b, loc, 16));
  Value ab = allocBank(b, loc, 1, 1), wb = allocBank(b, loc, 1, 1);
  Value db = allocBank(b, loc, 1, 1), cb = allocBank(b, loc, 1, 4);
  db = mvinBank(b, loc, dTile, db, 16);

  auto rowLoop =
      b.create<scf::ForOp>(loc, zero, idx(M), sixteen, ValueRange{ab, wb, cb});
  b.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();
  auto colLoop = b.create<scf::ForOp>(loc, zero, idx(N), sixteen,
                                      rowLoop.getRegionIterArgs());
  b.setInsertionPointToStart(colLoop.getBody());
  Value col = colLoop.getInductionVar();
  auto kLoop = b.create<scf::ForOp>(loc, zero, idx(K), sixteen,
                                    colLoop.getRegionIterArgs());
  b.setInsertionPointToStart(kLoop.getBody());
  Value k = kLoop.getInductionVar();
  ab = kLoop.getRegionIterArgs()[0];
  wb = kLoop.getRegionIterArgs()[1];
  cb = kLoop.getRegionIterArgs()[2];

  panel = b.create<arith::AddIOp>(
      loc,
      b.create<arith::MulIOp>(loc, b.create<arith::DivUIOp>(loc, row, sixteen),
                              idx(kTiles)),
      b.create<arith::DivUIOp>(loc, k, sixteen));
  SmallVector<OpFoldResult> panelOffsets{panel, b.getIndexAttr(0),
                                         b.getIndexAttr(0)};
  SmallVector<OpFoldResult> panelSizes{b.getIndexAttr(1), b.getIndexAttr(16),
                                       b.getIndexAttr(16)};
  SmallVector<OpFoldResult> panelSteps(3, b.getIndexAttr(1));
  auto panelType = memref::SubViewOp::inferRankReducedResultType(
      {16, 16}, cast<MemRefType>(panels.getType()), panelOffsets, panelSizes,
      panelSteps);
  Value aTile = b.create<memref::SubViewOp>(
      loc, panelType, panels, panelOffsets, panelSizes, panelSteps);
  ab = mvinBank(b, loc, aTile, ab, 16);

  Value full = b.create<arith::AndIOp>(
      loc, wDirect,
      b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(
              loc, P::ule, b.create<arith::AddIOp>(loc, k, sixteen), idx(K)),
          b.create<arith::CmpIOp>(loc, P::ule,
                                  b.create<arith::AddIOp>(loc, col, sixteen),
                                  idx(N))));
  auto weight = b.create<scf::IfOp>(loc, b.getI64Type(), full, true);
  b.setInsertionPointToStart(&weight.getThenRegion().front());
  SmallVector<OpFoldResult> wOffsets{k, col};
  SmallVector<OpFoldResult> wSizes(2, b.getIndexAttr(16));
  SmallVector<OpFoldResult> wSteps(2, b.getIndexAttr(1));
  Value wBlock = b.create<memref::SubViewOp>(loc, op.getWeight(), wOffsets,
                                             wSizes, wSteps);
  b.create<scf::YieldOp>(
      loc, ValueRange{b.create<BankMvinOp>(loc, b.getI64Type(), wBlock, wb,
                                           iter, wLines, IntegerAttr{})});
  b.setInsertionPointToStart(&weight.getElseRegion().front());
  auto wi = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(wi.getBody());
  auto wj = b.create<scf::ForOp>(loc, zero, sixteen, one);
  b.setInsertionPointToStart(wj.getBody());
  Value wk = b.create<arith::AddIOp>(loc, k, wi.getInductionVar());
  Value wn = b.create<arith::AddIOp>(loc, col, wj.getInductionVar());
  auto pick = b.create<scf::IfOp>(
      loc, b.getI8Type(),
      b.create<arith::AndIOp>(loc,
                              b.create<arith::CmpIOp>(loc, P::ult, wk, idx(K)),
                              b.create<arith::CmpIOp>(loc, P::ult, wn, idx(N))),
      true);
  b.setInsertionPointToStart(&pick.getThenRegion().front());
  b.create<scf::YieldOp>(loc, ValueRange{b.create<memref::LoadOp>(
                                  loc, op.getWeight(), ValueRange{wk, wn})});
  b.setInsertionPointToStart(&pick.getElseRegion().front());
  b.create<scf::YieldOp>(loc, zeroI8);
  b.setInsertionPointAfter(pick);
  b.create<memref::StoreOp>(
      loc, pick.getResult(0), wTile,
      ValueRange{wi.getInductionVar(), wj.getInductionVar()});
  b.setInsertionPointAfter(wi);
  b.create<scf::YieldOp>(loc, mvinBank(b, loc, wTile, wb, 16));
  b.setInsertionPointAfter(weight);
  wb = weight.getResult(0);

  auto first = b.create<arith::CmpIOp>(loc, P::eq, k, zero);
  auto compute = b.create<scf::IfOp>(loc, b.getI64Type(), first, true);
  b.setInsertionPointToStart(&compute.getThenRegion().front());
  Value preloaded = b.create<BankGemminiPreloadOp>(loc, b.getI64Type(), wb, cb,
                                                   iter, base, base);
  b.create<scf::YieldOp>(
      loc,
      ValueRange{b.create<BankGemminiComputePreloadedOp>(
          loc, b.getI64Type(), ab, db, preloaded, iter, base, base, base)});
  b.setInsertionPointToStart(&compute.getElseRegion().front());
  b.create<scf::YieldOp>(
      loc, ValueRange{b.create<BankGemminiComputeAccumulatedOp>(
               loc, b.getI64Type(), ab, wb, cb, iter, base, base, base)});
  b.setInsertionPointAfter(compute);
  b.create<scf::YieldOp>(loc, ValueRange{ab, wb, compute.getResult(0)});
  b.setInsertionPointAfter(kLoop);

  cb = mvoutBank(b, loc, acc, kLoop.getResult(2), 16);
  auto ei = b.create<scf::ForOp>(
      loc, zero,
      b.create<arith::MinUIOp>(loc, sixteen,
                               b.create<arith::SubIOp>(loc, idx(M), row)),
      one);
  b.setInsertionPointToStart(ei.getBody());
  auto ej = b.create<scf::ForOp>(
      loc, zero,
      b.create<arith::MinUIOp>(loc, sixteen,
                               b.create<arith::SubIOp>(loc, idx(N), col)),
      one);
  b.setInsertionPointToStart(ej.getBody());
  Value ri = b.create<arith::AddIOp>(loc, row, ei.getInductionVar());
  Value cj = b.create<arith::AddIOp>(loc, col, ej.getInductionVar());
  Value sum = b.create<memref::LoadOp>(
      loc, acc, ValueRange{ei.getInductionVar(), ej.getInductionVar()});
  Value result;
  if (integer) {
    result = b.create<arith::AddIOp>(
        loc, sum, b.create<memref::LoadOp>(loc, op.getBias(), ValueRange{cj}));
  } else {
    result = b.create<arith::AddFOp>(
        loc,
        b.create<arith::MulFOp>(
            loc, b.create<arith::SIToFPOp>(loc, b.getF32Type(), sum),
            b.create<arith::ConstantOp>(loc,
                                        b.getF32FloatAttr(inScale * wScale))),
        b.create<memref::LoadOp>(loc, op.getBias(), ValueRange{cj}));
  }
  b.create<memref::StoreOp>(
      loc, result, op.getOutput(),
      ValueRange{b.create<arith::DivUIOp>(loc, ri, idx(OH * OW)), cj,
                 b.create<arith::RemUIOp>(
                     loc, b.create<arith::DivUIOp>(loc, ri, idx(OW)), idx(OH)),
                 b.create<arith::RemUIOp>(loc, ri, idx(OW))});
  b.setInsertionPointAfter(ei);
  b.create<scf::YieldOp>(
      loc, ValueRange{kLoop.getResult(0), kLoop.getResult(1), cb});
  b.setInsertionPointAfter(colLoop);
  b.create<scf::YieldOp>(loc, colLoop.getResults());
  b.setInsertionPointAfter(rowLoop);

  for (Value bank : rowLoop.getResults())
    releaseBank(b, loc, bank);
  releaseBank(b, loc, db);
  for (Value buffer : {panels, dTile, wTile, acc})
    b.create<memref::DeallocOp>(loc, buffer);
  b.eraseOp(op);
  return success();
}
