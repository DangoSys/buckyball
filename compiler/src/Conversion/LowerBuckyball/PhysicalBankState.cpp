//===- PhysicalBankState.cpp - Physical bank allocation state -------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace mlir::buddy;
using namespace ::buddy::buckyball;

PhysicalBankState::PhysicalBankState(int64_t bankNum)
    : bankNum(bankNum), used(bankNum, 0) {}

std::optional<int64_t> PhysicalBankState::getConstI64(Value value) const {
  auto cst = value.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return std::nullopt;
  auto attr = dyn_cast<IntegerAttr>(cst.getValue());
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

std::optional<int64_t> PhysicalBankState::tryAlloc(int64_t row, int64_t col) {
  int64_t need = row * col;
  for (int64_t start = 0; start + need <= bankNum; ++start) {
    bool ok = true;
    for (int64_t i = 0; i < need; ++i) {
      if (used[start + i]) {
        ok = false;
        break;
      }
    }
    if (!ok)
      continue;
    for (int64_t i = 0; i < need; ++i)
      used[start + i] = 1;
    return start;
  }
  return std::nullopt;
}

LogicalResult PhysicalBankState::release(Operation *op, int64_t bank) {
  auto it = vm.find(bank);
  if (it == vm.end()) {
    op->emitError("release of unknown virtual bank handle");
    return failure();
  }
  freeAlloc(it->second);
  vm.erase(it);
  return success();
}

void PhysicalBankState::remember(int64_t bank, int64_t row, int64_t col) {
  vm[bank] = BankSlot{bank, row, col};
}

Value PhysicalBankState::cstI64(OpBuilder &builder, Location loc,
                                uint64_t value) const {
  return builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                           builder.getI64IntegerAttr(value));
}

void PhysicalBankState::createMset(OpBuilder &builder, Location loc,
                                   uint64_t bankId, bool alloc, uint64_t row,
                                   uint64_t col) const {
  auto op = builder.create<MsetOp>(loc, cstI64(builder, loc, bankId));
  op->setAttr("alloc", builder.getBoolAttr(alloc));
  op->setAttr("row", builder.getI64IntegerAttr(row));
  op->setAttr("col", builder.getI64IntegerAttr(col));
}

void PhysicalBankState::freeAlloc(const BankSlot &slot) {
  int64_t need = slot.row * slot.col;
  for (int64_t i = 0; i < need; ++i)
    used[slot.base + i] = 0;
}
