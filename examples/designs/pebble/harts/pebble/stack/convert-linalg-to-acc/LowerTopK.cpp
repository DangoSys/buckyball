//===- LowerTopK.cpp - Nonnegative score selection on integer keys -------===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "maxpoolOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult pebble::lowerTopK(func::FuncOp function) {
  SmallVector<scf::ExecuteRegionOp> operations;
  function.walk([&](scf::ExecuteRegionOp op) {
    if (op->hasAttr("pebble.topk_u32"))
      operations.push_back(op);
  });
  for (auto op : operations) {
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(op->getRegions(), captured);
    SmallVector<Value> scores;
    for (Value value : captured) {
      if (isa<RankedTensorType>(value.getType()))
        scores.push_back(value);
      else if (!value.getDefiningOp<arith::ConstantOp>())
        return op.emitOpError(
            "Top-K reference may capture only scores and scalar constants");
    }
    if (scores.size() != 1 || op.getNumResults() != 1)
      return op.emitOpError("Top-K region must capture one score tensor and "
                            "return one packed buffer");
    auto type = dyn_cast<RankedTensorType>(scores[0].getType());
    int64_t k = cast<IntegerAttr>(op->getAttr("pebble.topk_u32")).getInt();
    if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
        type.getDimSize(0) != 1 ||
        !(type.getElementType().isF32() ||
          type.getElementType().isInteger(32)) ||
        op.getResult(0).getType() !=
            MemRefType::get({(k + 1) / 2, 4},
                            IntegerType::get(op.getContext(), 32)))
      return op.emitOpError("Top-K requires one FP32 or U32 score row and "
                            "packed I32 key/index rows");
    int64_t n = type.getDimSize(1);
    const auto &target = target::getTarget();
    int64_t groups = llvm::divideCeil(n, target.bankDepth * 4);
    if (n <= 0 || k <= 0 || k > n || k > target.bankDepth * 2 ||
        groups + 1 >= target.bankNum)
      return op.emitOpError("Top-K input or output exceeds the bank geometry");
    int64_t rows = llvm::divideCeil(n, groups * 4);
    int64_t fullRows = n / (groups * 4);
    int64_t chunk = (target.dmaMaxRanges - 1) * 4096 / (groups * 16);
    if (chunk <= 0)
      return op.emitOpError(
          "Top-K group stride exceeds the prepared DMA range capacity");
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    Value zero = index(0), one = index(1);
    Value input = tensorBuffer(b, loc, scores[0]);
    Value source = BankAllocOp::create(b, loc, b.getI64Type(), 1, groups);
    if (n != rows * groups * 4)
      source = BankClearOp::create(b, loc, b.getI64Type(), source, 1, groups);
    for (int64_t group = 0; group < groups; ++group) {
      for (int64_t start = 0; start < fullRows; start += chunk) {
        Value view = memref::SubViewOp::create(
            b, loc, input,
            ArrayRef<OpFoldResult>{
                zero, b.getIndexAttr((start * groups + group) * 4)},
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
        source = BankMvinStridedOp::create(
            b, loc, b.getI64Type(), view, source, index(start),
            index(std::min(chunk, fullRows - start)), index(groups * 16), group,
            1, index(16), zero);
      }
      int64_t lanes = std::clamp(n - (fullRows * groups + group) * 4,
                                 int64_t(0), int64_t(4));
      if (lanes) {
        Value view = memref::SubViewOp::create(
            b, loc, input,
            ArrayRef<OpFoldResult>{
                zero, b.getIndexAttr((fullRows * groups + group) * 4)},
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
            SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
        source = BankMvinStridedOp::create(
            b, loc, b.getI64Type(), view, source, index(fullRows), one,
            index(groups * 16), group, 1, index(lanes * 4), zero);
      }
    }
    Value destination = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    destination = ::buddy::maxpool_ball::BankTopKU32Op::create(
        b, loc, b.getI64Type(), source, destination, rows, n, k);
    Value packed = memref::AllocOp::create(
        b, loc, MemRefType::get({(k + 1) / 2, 4}, b.getI32Type()));
    Value stored = BankMvoutOp::create(b, loc, b.getI64Type(), packed,
                                       destination, (k + 1) / 2, 1);
    BankReleaseOp::create(b, loc, source);
    BankReleaseOp::create(b, loc, stored);
    op.getResult(0).replaceAllUsesWith(packed);
    op.erase();
  }
  return success();
}
