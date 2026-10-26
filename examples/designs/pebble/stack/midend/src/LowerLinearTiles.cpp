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
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Type i64 = builder.getI64Type();
  auto index = [&](OpBuilder &b, int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  // Host loops over the valid part of a padded tile: min(size, limit - start).
  auto extent = [&](OpBuilder &b, int64_t size, int64_t limit,
                    Value start) -> Value {
    return arith::MinSIOp::create(
        b, loc, index(b, size),
        arith::SubIOp::create(b, loc, index(b, limit), start));
  };
  Value zero = index(builder, 0), one = index(builder, 1);
  Value input = tensorBuffer(builder, loc, args[0]);
  Value weight = tensorBuffer(builder, loc, args[1]);
  Value bias = tensorBuffer(builder, loc, args[2]);
  Value step;
  if (!integer)
    step = tensorBuffer(builder, loc, args[requant ? 3 : 5]);
  Value multiplier, inputStep;
  if (!requant && !integer) {
    multiplier = tensor::ExtractOp::create(builder, loc, args[3],
                                           ValueRange{zero, zero});
    inputStep = tensor::ExtractOp::create(builder, loc, args[4],
                                          ValueRange{zero, zero});
  }
  Value output = memref::AllocOp::create(
      builder, loc, MemRefType::get({m, n}, resultType.getElementType()));
  Value resultPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({rows, 16}, resultType.getElementType()));
  Value biasPack = memref::AllocOp::create(
      builder, loc, MemRefType::get({16}, builder.getI32Type()));
  Value scalePack;
  if (!integer)
    scalePack = memref::AllocOp::create(
        builder, loc, MemRefType::get({16}, builder.getF32Type()));

  pebble::repeat(
      builder, loc, 0, n, 16, false, {},
      [&](OpBuilder &b, Value col, ValueRange) {
        Value columns = extent(b, 16, n, col);
        linalg::FillOp::create(
            b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 0, 32)},
            biasPack);
        Value scales;
        if (!integer) {
          linalg::FillOp::create(b, loc,
                                 ValueRange{arith::ConstantOp::create(
                                     b, loc, b.getF32FloatAttr(1))},
                                 scalePack);
          auto lanes = scf::ForOp::create(b, loc, zero, columns, one);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(lanes.getBody());
            Value lane = lanes.getInductionVar();
            Value channel = arith::AddIOp::create(b, loc, col, lane);
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
        pebble::repeat(
            b, loc, 0, m, rows, false, {},
            [&](OpBuilder &b, Value row, ValueRange) {
              Value valid = extent(b, rows, m, row);
              if (integer) {
                auto lanes = scf::ForOp::create(b, loc, zero, columns, one);
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(lanes.getBody());
                Value lane = lanes.getInductionVar();
                Value channel = arith::AddIOp::create(b, loc, col, lane);
                Value value = memref::LoadOp::create(b, loc, bias,
                                                     ValueRange{row, channel});
                memref::StoreOp::create(b, loc, value, biasPack, lane);
              }
              Value accumulator = BankAllocOp::create(b, loc, i64, 1, 1);
              accumulator =
                  BankMvinOp::create(b, loc, i64, biasPack, accumulator, 4, 1);
              accumulator = pebble::accumulate(
                  b, loc, k, maxK, false, accumulator,
                  [&](OpBuilder &b, Value start, int64_t count, bool first,
                      bool last, Value accumulator) -> Value {
                    int64_t reduction = (count + 15) / 16 * 16;
                    Value inputPack = memref::AllocOp::create(
                        b, loc,
                        MemRefType::get({reduction / 16 * rows, 16},
                                        a.getElementType()));
                    Value weightPack = memref::AllocOp::create(
                        b, loc,
                        MemRefType::get({reduction, 16}, b.getI8Type()));
                    Value fzero =
                        arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
                    Value izero = arith::ConstantIntOp::create(b, loc, 0, 8);
                    linalg::FillOp::create(b, loc,
                                           (requant || integer) ? izero : fzero,
                                           inputPack);
                    linalg::FillOp::create(b, loc, izero, weightPack);
                    auto inner =
                        scf::ForOp::create(b, loc, zero, index(b, count), one);
                    {
                      OpBuilder::InsertionGuard guard(b);
                      b.setInsertionPointToStart(inner.getBody());
                      Value reductionIndex = inner.getInductionVar();
                      Value sourceK =
                          arith::AddIOp::create(b, loc, reductionIndex, start);
                      Value block = arith::MulIOp::create(
                          b, loc,
                          arith::DivUIOp::create(b, loc, reductionIndex,
                                                 index(b, 16)),
                          index(b, rows));
                      Value lane = arith::RemUIOp::create(
                          b, loc, reductionIndex, index(b, 16));
                      auto inputRows =
                          scf::ForOp::create(b, loc, zero, valid, one);
                      {
                        OpBuilder::InsertionGuard rowGuard(b);
                        b.setInsertionPointToStart(inputRows.getBody());
                        Value r = inputRows.getInductionVar();
                        Value sourceM = arith::AddIOp::create(b, loc, r, row);
                        Value destination =
                            arith::AddIOp::create(b, loc, block, r);
                        Value value = memref::LoadOp::create(
                            b, loc, input, ValueRange{sourceM, sourceK});
                        memref::StoreOp::create(b, loc, value, inputPack,
                                                ValueRange{destination, lane});
                      }
                      auto lanes =
                          scf::ForOp::create(b, loc, zero, columns, one);
                      b.setInsertionPointToStart(lanes.getBody());
                      Value c = lanes.getInductionVar();
                      Value sourceN = arith::AddIOp::create(b, loc, c, col);
                      Value value = memref::LoadOp::create(
                          b, loc, weight, ValueRange{sourceK, sourceN});
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
                      auto quant = BankCustomOp::create(
                          b, loc, i64, ValueRange{fp}, codes,
                          ValueRange{multiplier}, "fp32_to_int8");
                      quant->setAttr("rows",
                                     b.getI64IntegerAttr(rows * reduction / 4));
                      codes = quant.getDestinationOut();
                      BankReleaseOp::create(b, loc, fp);
                    }
                    Value weights = BankAllocOp::create(b, loc, i64, 1, 1);
                    weights = BankMvinOp::create(b, loc, i64, weightPack,
                                                 weights, reduction, 1);
                    auto matmul = BankCustomOp::create(
                        b, loc, i64, ValueRange{codes, weights}, accumulator,
                        ValueRange{}, "matmul_i8_block", true);
                    matmul->setAttr("rows", b.getI64IntegerAttr(rows));
                    matmul->setAttr("k", b.getI64IntegerAttr(reduction));
                    matmul->setAttr("first", b.getBoolAttr(first));
                    matmul->setAttr("last", b.getBoolAttr(last));
                    BankReleaseOp::create(b, loc, codes);
                    BankReleaseOp::create(b, loc, weights);
                    return matmul.getDestinationOut();
                  });
              Value restored = accumulator;
              if (!integer) {
                restored = BankAllocOp::create(b, loc, i64, 1, 1);
                auto dequant = BankCustomOp::create(
                    b, loc, i64, ValueRange{accumulator, scales}, restored,
                    ValueRange{}, requant ? "int32_to_int8" : "int32_to_fp32");
                dequant->setAttr("rows", b.getI64IntegerAttr(rows));
                dequant->setAttr("relu", requant ? op->getAttr("relu")
                                                 : b.getBoolAttr(false));
                restored = dequant.getDestinationOut();
              }
              BankMvoutOp::create(b, loc, i64, resultPack, restored,
                                  rows * (requant ? 1 : 4), 1);
              BankReleaseOp::create(b, loc, accumulator);
              if (!integer)
                BankReleaseOp::create(b, loc, restored);
              auto outputRows = scf::ForOp::create(b, loc, zero, valid, one);
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(outputRows.getBody());
              Value r = outputRows.getInductionVar();
              auto lanes = scf::ForOp::create(b, loc, zero, columns, one);
              b.setInsertionPointToStart(lanes.getBody());
              Value c = lanes.getInductionVar();
              Value value =
                  memref::LoadOp::create(b, loc, resultPack, ValueRange{r, c});
              memref::StoreOp::create(
                  b, loc, value, output,
                  ValueRange{arith::AddIOp::create(b, loc, r, row),
                             arith::AddIOp::create(b, loc, c, col)});
              return SmallVector<Value>{};
            });
        if (!integer)
          BankReleaseOp::create(b, loc, scales);
        return SmallVector<Value>{};
      });
  result = bufferization::ToTensorOp::create(builder, loc, resultType, output,
                                             true, true);
  return success();
}
