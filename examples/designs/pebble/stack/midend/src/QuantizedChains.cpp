//===- QuantizedChains.cpp - Resident INT8 vector and channel chains ------===//
#include "Buckyball/BuckyballOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using Computation = ::buddy::custom::CustomOp;

void pebble::formQuantizedChains(func::FuncOp function) {
  SmallVector<Computation> roots;
  function.walk([&](Computation op) {
    if (op.getKind() == "global_avg_i8" ||
        (op.getKind() == "linear_i8_requant" &&
         cast<RankedTensorType>(op.getResult(0).getType()).getDimSize(0) == 1))
      roots.push_back(op);
  });
  int64_t id = 0;
  for (auto root : roots) {
    if (root->hasAttr("custom.quantized_chain"))
      continue;
    SmallVector<Operation *> chain{root};
    Value value = root.getResult(0);
    int computations = 1;
    while (value.hasOneUse()) {
      Operation *user = *value.getUsers().begin();
      if (user->getBlock() != root->getBlock())
        break;
      bool barrier = false;
      for (Operation *between = chain.back()->getNextNode(); between != user;
           between = between->getNextNode())
        if (!isMemoryEffectFree(between)) {
          barrier = true;
          break;
        }
      if (barrier)
        break;
      if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(user)) {
        auto type = collapse.getResultType();
        if (type.getRank() != 2 || type.getDimSize(0) != 1)
          break;
      } else if (auto op = dyn_cast<Computation>(user)) {
        if (op.getInputs()[0] != value || op->hasAttr("custom.quantized_chain"))
          break;
        if (op.getKind() == "gate_mul_i8") {
          if (!op.getResult(0).hasOneUse())
            break;
          auto consumer =
              dyn_cast<Computation>(*op.getResult(0).getUsers().begin());
          auto type = cast<RankedTensorType>(op.getInputs()[1].getType());
          if (!consumer || consumer.getKind() != "linear_i8_requant" ||
              consumer.getInputs()[0] != op.getResult(0) ||
              type.getDimSize(1) % 16 ||
              consumer->getBlock() != root->getBlock())
            break;
          for (Operation *between = op->getNextNode(); between != consumer;
               between = between->getNextNode())
            if (!isMemoryEffectFree(between)) {
              barrier = true;
              break;
            }
          if (barrier)
            break;
          chain.append({op, consumer});
          computations += 2;
          break;
        }
        if (op.getKind() == "lookup_i8" &&
            cast<RankedTensorType>(op.getInputs()[1].getType())
                    .getNumElements() != 256)
          break;
        if ((op.getKind() != "linear_i8_requant" &&
             op.getKind() != "lookup_i8") ||
            cast<RankedTensorType>(op.getResult(0).getType()).getDimSize(0) !=
                1)
          break;
        ++computations;
      } else {
        break;
      }
      chain.push_back(user);
      value = user->getResult(0);
    }
    if (computations < 2)
      continue;
    for (Operation *op : chain)
      op->setAttr(
          "custom.quantized_chain",
          IntegerAttr::get(IntegerType::get(function.getContext(), 64), id));
    ++id;
  }
}

namespace {
LogicalResult vectorLinear(Computation op, Value input, Value &result) {
  auto args = op.getInputs();
  auto weightType = cast<RankedTensorType>(args[1].getType());
  int64_t k = weightType.getDimSize(0), n = weightType.getDimSize(1);
  int64_t paddedK = (k + 15) / 16 * 16;
  if (paddedK > buckyball_target::getBuckyballTarget().bankDepth || n > 2048)
    return op.emitOpError(
        "resident Linear exceeds its bank or quantizer row range");
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(0), one = index(1);
  Type i64 = b.getI64Type();
  Value weights = pebble::tensorBuffer(b, loc, args[1]);
  Value biases = pebble::tensorBuffer(b, loc, args[2]);
  Value scales = pebble::tensorBuffer(b, loc, args[3]);
  result = BankAllocOp::create(b, loc, i64, 1, 1);
  for (int64_t c = 0; c < n; c += 16) {
    Value weightPack = memref::AllocOp::create(
        b, loc, MemRefType::get({paddedK, 16}, b.getI8Type()));
    Value biasPack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getI32Type()));
    Value scalePack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getF32Type()));
    Value zeroI8 = arith::ConstantIntOp::create(b, loc, 0, 8);
    Value zeroI32 = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value oneF32 = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1));
    linalg::FillOp::create(b, loc, zeroI8, weightPack);
    linalg::FillOp::create(b, loc, zeroI32, biasPack);
    linalg::FillOp::create(b, loc, oneF32, scalePack);
    auto lanes = scf::ForOp::create(b, loc, zero,
                                    index(std::min<int64_t>(16, n - c)), one);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(lanes.getBody());
      Value lane = lanes.getInductionVar();
      Value channel = arith::AddIOp::create(b, loc, lane, index(c));
      Value bias =
          memref::LoadOp::create(b, loc, biases, ValueRange{zero, channel});
      Value scale =
          memref::LoadOp::create(b, loc, scales, ValueRange{zero, channel});
      memref::StoreOp::create(b, loc, bias, biasPack, lane);
      memref::StoreOp::create(b, loc, scale, scalePack, lane);
      auto inner = scf::ForOp::create(b, loc, zero, index(k), one);
      b.setInsertionPointToStart(inner.getBody());
      Value r = inner.getInductionVar();
      Value weight =
          memref::LoadOp::create(b, loc, weights, ValueRange{r, channel});
      memref::StoreOp::create(b, loc, weight, weightPack, ValueRange{r, lane});
    }
    Value w = BankAllocOp::create(b, loc, i64, 1, 1);
    w = BankMvinOp::create(b, loc, i64, weightPack, w, paddedK, 1);
    Value acc = BankAllocOp::create(b, loc, i64, 1, 1);
    acc = BankMvinOp::create(b, loc, i64, biasPack, acc, 4, 1);
    auto product = BankCustomOp::create(b, loc, i64, ValueRange{input, w}, acc,
                                        ValueRange{}, "matmul_i8", true);
    product->setAttr("rows", b.getI64IntegerAttr(1));
    product->setAttr("k", b.getI64IntegerAttr(paddedK));
    acc = product.getDestinationOut();
    BankReleaseOp::create(b, loc, w);
    Value scale = BankAllocOp::create(b, loc, i64, 1, 1);
    scale = BankMvinOp::create(b, loc, i64, scalePack, scale, 4, 1);
    auto quant =
        BankCustomOp::create(b, loc, i64, ValueRange{acc, scale}, result,
                             ValueRange{}, "int32_to_int8", true);
    quant->setAttr("rows", b.getI64IntegerAttr(1));
    quant->setAttr("wr_base", b.getI64IntegerAttr(c / 16));
    quant->setAttr("relu", op->getAttr("relu"));
    result = quant.getDestinationOut();
    BankReleaseOp::create(b, loc, acc);
    BankReleaseOp::create(b, loc, scale);
  }
  BankReleaseOp::create(b, loc, input);
  return success();
}

LogicalResult gatedLinear(Computation product, Computation multiply, Value gate,
                          Value &result) {
  auto args = product.getInputs();
  auto inputType = cast<RankedTensorType>(multiply.getInputs()[1].getType());
  auto resultType = cast<RankedTensorType>(product.getResult(0).getType());
  int64_t m = inputType.getDimSize(0), k = inputType.getDimSize(1),
          n = resultType.getDimSize(1);
  int64_t rows =
      buckyball_target::getBuckyballBallParam("SMATMUL_OS", "tileRows");
  int64_t reductionLimit =
      std::min<int64_t>(64 * 16 / rows,
                        buckyball_target::getBuckyballTarget().bankDepth) /
      16 * 16;
  OpBuilder b(product);
  Location loc = product.getLoc();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(0), one = index(1);
  Type i64 = b.getI64Type();
  Value input = pebble::tensorBuffer(b, loc, multiply.getInputs()[1]);
  Value weights = pebble::tensorBuffer(b, loc, args[1]);
  Value biases = pebble::tensorBuffer(b, loc, args[2]);
  Value scales = pebble::tensorBuffer(b, loc, args[3]);
  Value ratio = arith::ConstantOp::create(
      b, loc, cast<FloatAttr>(multiply->getAttr("ratio")));
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(resultType.getShape(), b.getI8Type()));
  Value scratch = memref::AllocOp::create(
      b, loc, MemRefType::get({rows, 16}, b.getI8Type()));
  Value izero = arith::ConstantIntOp::create(b, loc, 0, 8);
  for (int64_t c = 0; c < n; c += 16) {
    Value biasPack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getI32Type()));
    Value scalePack =
        memref::AllocOp::create(b, loc, MemRefType::get({16}, b.getF32Type()));
    Value zeroI32 = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value oneF32 = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1));
    linalg::FillOp::create(b, loc, zeroI32, biasPack);
    linalg::FillOp::create(b, loc, oneF32, scalePack);
    auto lanes = scf::ForOp::create(b, loc, zero,
                                    index(std::min<int64_t>(16, n - c)), one);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(lanes.getBody());
      Value lane = lanes.getInductionVar(),
            channel = arith::AddIOp::create(b, loc, lane, index(c));
      Value bias =
          memref::LoadOp::create(b, loc, biases, ValueRange{zero, channel});
      Value scale =
          memref::LoadOp::create(b, loc, scales, ValueRange{zero, channel});
      memref::StoreOp::create(b, loc, bias, biasPack, lane);
      memref::StoreOp::create(b, loc, scale, scalePack, lane);
    }
    Value scale = BankAllocOp::create(b, loc, i64, 1, 1);
    scale = BankMvinOp::create(b, loc, i64, scalePack, scale, 4, 1);
    for (int64_t r = 0; r < m; r += rows) {
      Value acc = BankAllocOp::create(b, loc, i64, 1, 1);
      acc = BankMvinOp::create(b, loc, i64, biasPack, acc, 4, 1);
      for (int64_t start = 0; start < k; start += reductionLimit) {
        int64_t count = std::min(reductionLimit, k - start);
        Value a = BankAllocOp::create(b, loc, i64, 1, 1);
        for (int64_t channel = start; channel < start + count; channel += 16) {
          linalg::FillOp::create(b, loc, izero, scratch);
          auto pixels = scf::ForOp::create(b, loc, zero,
                                           index(std::min(rows, m - r)), one);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(pixels.getBody());
            Value row = pixels.getInductionVar();
            auto lanes = scf::ForOp::create(b, loc, zero, index(16), one);
            b.setInsertionPointToStart(lanes.getBody());
            Value lane = lanes.getInductionVar();
            Value value = memref::LoadOp::create(
                b, loc, input,
                ValueRange{
                    arith::AddIOp::create(b, loc, row, index(r)),
                    arith::AddIOp::create(b, loc, lane, index(channel))});
            memref::StoreOp::create(b, loc, value, scratch,
                                    ValueRange{row, lane});
          }
          Value source = BankAllocOp::create(b, loc, i64, 1, 1);
          source = BankMvinOp::create(b, loc, i64, scratch, source, rows, 1);
          Value gated = BankAllocOp::create(b, loc, i64, 1, 1);
          auto mul =
              BankCustomOp::create(b, loc, i64, ValueRange{gate, source}, gated,
                                   ValueRange{ratio}, "gate_mul_i8");
          mul->setAttr("rows", b.getI64IntegerAttr(rows));
          mul->setAttr("gate_row", b.getI64IntegerAttr(channel / 16));
          gated = mul.getDestinationOut();
          BankReleaseOp::create(b, loc, source);
          for (int64_t offset = 0; offset < rows; offset += 4) {
            auto copy = BankCustomOp::create(b, loc, i64, ValueRange{gated}, a,
                                             ValueRange{}, "maxpool_i8", true);
            for (auto [name, value] :
                 std::initializer_list<std::pair<StringRef, int64_t>>{
                     {"input_side", 2},
                     {"output_side", 2},
                     {"kernel", 1},
                     {"stride", 1},
                     {"padding", 0},
                     {"op1_base", offset},
                     {"wr_base", (channel - start) / 16 * rows + offset},
                     {"output_stride", 2},
                     {"start_row", 0},
                     {"start_col", 0}})
              copy->setAttr(name, b.getI64IntegerAttr(value));
            a = copy.getDestinationOut();
          }
          BankReleaseOp::create(b, loc, gated);
        }
        Value weightPack = memref::AllocOp::create(
            b, loc, MemRefType::get({count, 16}, b.getI8Type()));
        linalg::FillOp::create(b, loc, izero, weightPack);
        auto inner = scf::ForOp::create(b, loc, zero, index(count), one);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(inner.getBody());
          Value row = inner.getInductionVar();
          auto lanes = scf::ForOp::create(
              b, loc, zero, index(std::min<int64_t>(16, n - c)), one);
          b.setInsertionPointToStart(lanes.getBody());
          Value lane = lanes.getInductionVar();
          Value value = memref::LoadOp::create(
              b, loc, weights,
              ValueRange{arith::AddIOp::create(b, loc, row, index(start)),
                         arith::AddIOp::create(b, loc, lane, index(c))});
          memref::StoreOp::create(b, loc, value, weightPack,
                                  ValueRange{row, lane});
        }
        Value w = BankAllocOp::create(b, loc, i64, 1, 1);
        w = BankMvinOp::create(b, loc, i64, weightPack, w, count, 1);
        auto matmul =
            BankCustomOp::create(b, loc, i64, ValueRange{a, w}, acc,
                                 ValueRange{}, "matmul_i8_block", true);
        matmul->setAttr("rows", b.getI64IntegerAttr(rows));
        matmul->setAttr("k", b.getI64IntegerAttr(count));
        matmul->setAttr("first", b.getBoolAttr(start == 0));
        matmul->setAttr("last", b.getBoolAttr(start + count == k));
        acc = matmul.getDestinationOut();
        BankReleaseOp::create(b, loc, a);
        BankReleaseOp::create(b, loc, w);
      }
      Value codes = BankAllocOp::create(b, loc, i64, 1, 1);
      auto quant = BankCustomOp::create(b, loc, i64, ValueRange{acc, scale},
                                        codes, ValueRange{}, "int32_to_int8");
      quant->setAttr("rows", b.getI64IntegerAttr(rows));
      quant->setAttr("wr_base", b.getI64IntegerAttr(0));
      quant->setAttr("relu", product->getAttr("relu"));
      codes = quant.getDestinationOut();
      BankMvoutOp::create(b, loc, i64, scratch, codes, rows, 1);
      BankReleaseOp::create(b, loc, acc);
      BankReleaseOp::create(b, loc, codes);
      auto pixels =
          scf::ForOp::create(b, loc, zero, index(std::min(rows, m - r)), one);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(pixels.getBody());
      Value row = pixels.getInductionVar();
      auto lanes = scf::ForOp::create(b, loc, zero,
                                      index(std::min<int64_t>(16, n - c)), one);
      b.setInsertionPointToStart(lanes.getBody());
      Value lane = lanes.getInductionVar();
      Value value =
          memref::LoadOp::create(b, loc, scratch, ValueRange{row, lane});
      memref::StoreOp::create(
          b, loc, value, output,
          ValueRange{arith::AddIOp::create(b, loc, row, index(r)),
                     arith::AddIOp::create(b, loc, lane, index(c))});
    }
    BankReleaseOp::create(b, loc, scale);
  }
  BankReleaseOp::create(b, loc, gate);
  result =
      bufferization::ToTensorOp::create(b, loc, resultType, output, true, true);
  return success();
}
} // namespace

LogicalResult pebble::lowerQuantizedChains(func::FuncOp function) {
  llvm::MapVector<int64_t, SmallVector<Operation *>> groups;
  function.walk([&](Operation *op) {
    if (auto id = op->getAttrOfType<IntegerAttr>("custom.quantized_chain"))
      groups[id.getInt()].push_back(op);
  });
  for (auto &[id, chain] : groups) {
    Value bank, result;
    Computation multiply;
    for (Operation *link : chain) {
      auto op = dyn_cast<Computation>(link);
      if (!op)
        continue;
      if (op.getKind() == "global_avg_i8") {
        if (failed(lowerGlobalAverage(op, bank, true)))
          return failure();
      } else if (op.getKind() == "linear_i8_requant") {
        if (multiply) {
          if (failed(gatedLinear(op, multiply, bank, result)))
            return failure();
          bank = Value();
        } else {
          if (!bank) {
            OpBuilder b(op);
            Location loc = op.getLoc();
            auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
            int64_t k = type.getDimSize(1), padded = (k + 15) / 16 * 16;
            Value pack = memref::AllocOp::create(
                b, loc, MemRefType::get({1, padded}, b.getI8Type()));
            Value zero = arith::ConstantIntOp::create(b, loc, 0, 8);
            linalg::FillOp::create(b, loc, zero, pack);
            Value slice = memref::SubViewOp::create(
                b, loc, pack, ArrayRef<int64_t>{0, 0}, ArrayRef<int64_t>{1, k},
                ArrayRef<int64_t>{1, 1});
            memref::CopyOp::create(
                b, loc, tensorBuffer(b, loc, op.getInputs()[0]), slice);
            bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            bank = BankMvinOp::create(b, loc, b.getI64Type(), pack, bank,
                                      padded / 16, 1);
          }
          Value next;
          if (failed(vectorLinear(op, bank, next)))
            return failure();
          bank = next;
        }
      } else if (op.getKind() == "lookup_i8") {
        OpBuilder b(op);
        Location loc = op.getLoc();
        auto tableType = cast<RankedTensorType>(op.getInputs()[1].getType());
        if (tableType.getNumElements() != 256)
          return op.emitOpError(
              "resident lookup requires one shared 256-entry table");
        Value table = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        table = BankMvinOp::create(b, loc, b.getI64Type(),
                                   tensorBuffer(b, loc, op.getInputs()[1]),
                                   table, 16, 1);
        Value destination = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        auto lookup = BankCustomOp::create(b, loc, b.getI64Type(),
                                           ValueRange{bank, table}, destination,
                                           ValueRange{}, "lookup_i8");
        int64_t channels =
            cast<RankedTensorType>(op.getResult(0).getType()).getDimSize(1);
        lookup->setAttr("rows", b.getI64IntegerAttr((channels + 15) / 16));
        BankReleaseOp::create(b, loc, bank);
        BankReleaseOp::create(b, loc, table);
        bank = lookup.getDestinationOut();
      } else if (op.getKind() == "gate_mul_i8") {
        multiply = op;
      }
    }
    Operation *last = chain.back();
    if (bank) {
      OpBuilder b(last);
      Location loc = last->getLoc();
      auto type = cast<RankedTensorType>(last->getResult(0).getType());
      int64_t channels = type.getDimSize(1), padded = (channels + 15) / 16 * 16;
      Value pack = memref::AllocOp::create(
          b, loc, MemRefType::get({1, padded}, b.getI8Type()));
      BankMvoutOp::create(b, loc, b.getI64Type(), pack, bank, padded / 16, 1);
      BankReleaseOp::create(b, loc, bank);
      Value output = memref::AllocOp::create(
          b, loc, MemRefType::get({1, channels}, b.getI8Type()));
      Value slice = memref::SubViewOp::create(
          b, loc, pack, ArrayRef<int64_t>{0, 0}, ArrayRef<int64_t>{1, channels},
          ArrayRef<int64_t>{1, 1});
      memref::CopyOp::create(b, loc, slice, output);
      result =
          bufferization::ToTensorOp::create(b, loc, type, output, true, true);
    }
    last->getResult(0).replaceAllUsesWith(result);
    for (Operation *op : llvm::reverse(chain))
      op->erase();
  }
  return success();
}
