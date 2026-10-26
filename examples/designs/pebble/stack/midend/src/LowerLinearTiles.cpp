//===- LowerLinearTiles.cpp - Linear rows and reduction/output tails ------===//
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

LogicalResult pebble::lowerLinearTiles(::buddy::custom::CustomOp op,
                                       Value &result) {
  bool integer = op.getKind() == "matmul_i8";
  bool requant = op.getKind() == "linear_i8_requant";
  auto args = op.getInputs();
  auto a = cast<RankedTensorType>(args[0].getType());
  auto w = cast<RankedTensorType>(args[1].getType());
  auto biasType = cast<RankedTensorType>(args[2].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (!a.hasStaticShape() || !w.hasStaticShape() || a.getRank() != 2 ||
      w.getRank() != 2 || a.getDimSize(1) != w.getDimSize(0) ||
      !((requant || integer) ? a.getElementType().isInteger(8)
                             : a.getElementType().isF32()) ||
      !w.getElementType().isInteger(8) ||
      !(integer   ? resultType.getElementType().isInteger(32)
        : requant ? resultType.getElementType().isInteger(8)
                  : resultType.getElementType().isF32()) ||
      !biasType.getElementType().isInteger(32) ||
      biasType.getShape() !=
          ArrayRef<int64_t>({integer ? a.getDimSize(0) : 1, w.getDimSize(1)}) ||
      resultType.getShape() !=
          ArrayRef<int64_t>({a.getDimSize(0), w.getDimSize(1)}))
    return op.emitOpError(
        "requires matching MxK/KxN tensors and INT32 1xN bias");
  int64_t m = a.getDimSize(0), k = a.getDimSize(1), n = w.getDimSize(1);
  const auto &target = buckyball_target::getBuckyballTarget();
  int64_t rows = integer ? 1
                         : buckyball_target::getBuckyballBallParam("SMATMUL_OS",
                                                                   "tileRows");
  int64_t maxK =
      std::min(target.bankDepth,
               target.bankDepth * ((requant || integer) ? 16 : 4) / rows) /
      16 * 16;
  if (m <= 0 || k <= 0 || n <= 0 || target.bankWidthBits != 128 || maxK == 0 ||
      rows * 4 > target.bankDepth)
    return op.emitOpError("Linear tile does not fit the configured banks");
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(0), one = index(1);
  Type i64 = b.getI64Type();
  Value input = tensorBuffer(b, loc, args[0]);
  Value weight = tensorBuffer(b, loc, args[1]);
  Value bias = tensorBuffer(b, loc, args[2]);
  Value step;
  if (!integer)
    step = tensorBuffer(b, loc, args[requant ? 3 : 5]);
  Value multiplier, inputStep;
  if (!requant && !integer) {
    multiplier =
        tensor::ExtractOp::create(b, loc, args[3], ValueRange{zero, zero});
    inputStep =
        tensor::ExtractOp::create(b, loc, args[4], ValueRange{zero, zero});
  }
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get({m, n}, resultType.getElementType()));
  Value resultPack = memref::AllocOp::create(
      b, loc, MemRefType::get({rows, 16}, resultType.getElementType()));
  for (int64_t col = 0; col < n; col += 16) {
    Value biasPack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getI32Type()));
    Value scalePack;
    if (!integer)
      scalePack = memref::AllocOp::create(
          b, loc, MemRefType::get({16}, b.getF32Type()));
    Value biasZero = arith::ConstantIntOp::create(b, loc, 0, 32);
    linalg::FillOp::create(b, loc, biasZero, biasPack);
    Value scales;
    if (!integer) {
      Value scaleOne = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1));
      linalg::FillOp::create(b, loc, scaleOne, scalePack);
      auto lanes = scf::ForOp::create(
          b, loc, zero, index(std::min<int64_t>(16, n - col)), one);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(lanes.getBody());
        Value lane = lanes.getInductionVar();
        Value channel = arith::AddIOp::create(b, loc, index(col), lane);
        Value value =
            memref::LoadOp::create(b, loc, bias, ValueRange{zero, channel});
        memref::StoreOp::create(b, loc, value, biasPack, lane);
        Value scale =
            memref::LoadOp::create(b, loc, step, ValueRange{zero, channel});
        if (!requant)
          scale = arith::MulFOp::create(b, loc, inputStep, scale);
        memref::StoreOp::create(b, loc, scale, scalePack, lane);
      }
      scales = BankAllocOp::create(b, loc, i64, 1, 1);
      scales = BankMvinOp::create(b, loc, i64, scalePack, scales, 4, 1);
    }
    for (int64_t row = 0; row < m; row += rows) {
      if (integer) {
        auto lanes = scf::ForOp::create(
            b, loc, zero, index(std::min<int64_t>(16, n - col)), one);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(lanes.getBody());
        Value lane = lanes.getInductionVar();
        Value channel = arith::AddIOp::create(b, loc, index(col), lane);
        Value value = memref::LoadOp::create(b, loc, bias,
                                             ValueRange{index(row), channel});
        memref::StoreOp::create(b, loc, value, biasPack, lane);
      }
      Value accumulator = BankAllocOp::create(b, loc, i64, 1, 1);
      accumulator =
          BankMvinOp::create(b, loc, i64, biasPack, accumulator, 4, 1);
      for (int64_t start = 0; start < k; start += maxK) {
        int64_t valid = std::min(maxK, k - start),
                reduction = (valid + 15) / 16 * 16;
        Value inputPack = memref::AllocOp::create(
            b, loc,
            MemRefType::get({reduction / 16 * rows, 16}, a.getElementType()));
        Value weightPack = memref::AllocOp::create(
            b, loc, MemRefType::get({reduction, 16}, b.getI8Type()));
        Value fzero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
        Value izero = arith::ConstantIntOp::create(b, loc, 0, 8);
        linalg::FillOp::create(b, loc, (requant || integer) ? izero : fzero,
                               inputPack);
        linalg::FillOp::create(b, loc, izero, weightPack);
        auto inner = scf::ForOp::create(b, loc, zero, index(valid), one);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(inner.getBody());
          Value reductionIndex = inner.getInductionVar();
          Value sourceK =
              arith::AddIOp::create(b, loc, reductionIndex, index(start));
          Value block = arith::MulIOp::create(
              b, loc, arith::DivUIOp::create(b, loc, reductionIndex, index(16)),
              index(rows));
          Value lane =
              arith::RemUIOp::create(b, loc, reductionIndex, index(16));
          auto inputRows = scf::ForOp::create(
              b, loc, zero, index(std::min(rows, m - row)), one);
          {
            OpBuilder::InsertionGuard rowGuard(b);
            b.setInsertionPointToStart(inputRows.getBody());
            Value r = inputRows.getInductionVar();
            Value sourceM = arith::AddIOp::create(b, loc, r, index(row));
            Value destination = arith::AddIOp::create(b, loc, block, r);
            Value value = memref::LoadOp::create(b, loc, input,
                                                 ValueRange{sourceM, sourceK});
            memref::StoreOp::create(b, loc, value, inputPack,
                                    ValueRange{destination, lane});
          }
          auto columns = scf::ForOp::create(
              b, loc, zero, index(std::min<int64_t>(16, n - col)), one);
          b.setInsertionPointToStart(columns.getBody());
          Value c = columns.getInductionVar();
          Value sourceN = arith::AddIOp::create(b, loc, c, index(col));
          Value value = memref::LoadOp::create(b, loc, weight,
                                               ValueRange{sourceK, sourceN});
          memref::StoreOp::create(b, loc, value, weightPack,
                                  ValueRange{reductionIndex, c});
        }
        Value codes = BankAllocOp::create(b, loc, i64, 1, 1);
        if (requant || integer) {
          codes = BankMvinOp::create(b, loc, i64, inputPack, codes,
                                     rows * reduction / 16, 1);
        } else {
          Value fp = BankAllocOp::create(b, loc, i64, 1, 1);
          fp = BankMvinOp::create(b, loc, i64, inputPack, fp,
                                  rows * reduction / 4, 1);
          auto quant =
              BankCustomOp::create(b, loc, i64, ValueRange{fp}, codes,
                                   ValueRange{multiplier}, "fp32_to_int8");
          quant->setAttr("rows", b.getI64IntegerAttr(rows * reduction / 4));
          codes = quant.getDestinationOut();
          BankReleaseOp::create(b, loc, fp);
        }
        Value weights = BankAllocOp::create(b, loc, i64, 1, 1);
        weights =
            BankMvinOp::create(b, loc, i64, weightPack, weights, reduction, 1);
        auto matmul = BankCustomOp::create(
            b, loc, i64, ValueRange{codes, weights}, accumulator, ValueRange{},
            "matmul_i8_block", true);
        matmul->setAttr("rows", b.getI64IntegerAttr(rows));
        matmul->setAttr("k", b.getI64IntegerAttr(reduction));
        matmul->setAttr("first", b.getBoolAttr(start == 0));
        matmul->setAttr("last", b.getBoolAttr(start + valid == k));
        accumulator = matmul.getDestinationOut();
        BankReleaseOp::create(b, loc, codes);
        BankReleaseOp::create(b, loc, weights);
      }
      Value restored = accumulator;
      if (!integer) {
        restored = BankAllocOp::create(b, loc, i64, 1, 1);
        auto dequant = BankCustomOp::create(
            b, loc, i64, ValueRange{accumulator, scales}, restored,
            ValueRange{}, requant ? "int32_to_int8" : "int32_to_fp32");
        dequant->setAttr("rows", b.getI64IntegerAttr(rows));
        if (requant) {
          dequant->setAttr("wr_base", b.getI64IntegerAttr(0));
          dequant->setAttr("relu", op->getAttr("relu"));
        }
        restored = dequant.getDestinationOut();
      }
      BankMvoutOp::create(b, loc, i64, resultPack, restored,
                          rows * (requant ? 1 : 4), 1);
      BankReleaseOp::create(b, loc, accumulator);
      if (!integer)
        BankReleaseOp::create(b, loc, restored);
      auto outputRows =
          scf::ForOp::create(b, loc, zero, index(std::min(rows, m - row)), one);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(outputRows.getBody());
      Value r = outputRows.getInductionVar();
      auto columns = scf::ForOp::create(
          b, loc, zero, index(std::min<int64_t>(16, n - col)), one);
      b.setInsertionPointToStart(columns.getBody());
      Value c = columns.getInductionVar();
      Value value =
          memref::LoadOp::create(b, loc, resultPack, ValueRange{r, c});
      memref::StoreOp::create(
          b, loc, value, output,
          ValueRange{arith::AddIOp::create(b, loc, r, index(row)),
                     arith::AddIOp::create(b, loc, c, index(col))});
    }
    if (!integer)
      BankReleaseOp::create(b, loc, scales);
  }
  result =
      bufferization::ToTensorOp::create(b, loc, resultType, output, true, true);
  return success();
}
