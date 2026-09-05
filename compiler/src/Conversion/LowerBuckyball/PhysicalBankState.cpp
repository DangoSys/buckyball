//===- PhysicalBankState.cpp - Physical bank allocation state -------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::buddy;
using namespace ::buddy::buckyball;

PhysicalBankState::PhysicalBankState(int64_t bankNum)
    : bankNum(bankNum), used(bankNum, 0) {}

int64_t PhysicalBankState::getUsedCount() const {
  return llvm::count(used, static_cast<int8_t>(1));
}

std::optional<int64_t> PhysicalBankState::getConstI64(Value value) const {
  while (true) {
    if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
      auto attr = dyn_cast<IntegerAttr>(cst.getValue());
      if (!attr)
        return std::nullopt;
      return attr.getInt();
    }
    if (auto op = value.getDefiningOp<BankMvinOp>()) {
      value = op.getBank();
      continue;
    }
    if (auto op = value.getDefiningOp<BankMvoutOp>()) {
      value = op.getBank();
      continue;
    }
    // Optional Balls: use op names so Cores can omit generated C++ types.
    if (Operation *op = value.getDefiningOp()) {
      StringRef name = op->getName().getStringRef();
      if (name == "buckyball.bank_transpose" ||
          name == "buckyball.bank_quant_f32_to_i8") {
        value = op->getOperand(1);
        continue;
      }
      if (name == "buckyball.bank_quant_i32_to_i8") {
        value = op->getOperand(2);
        continue;
      }
      if (name == "buckyball.bank_int32_to_fp32") {
        value = op->getOperand(2);
        continue;
      }
      if (name == "buckyball.bank_smatmul_bias") {
        value = op->getOperand(0);
        continue;
      }
      if (name == "buckyball.bank_im2col") {
        value = op->getOperand(1);
        continue;
      }
      if (name == "buckyball.bank_lut") {
        value = op->getOperand(2);
        continue;
      }
      if (name == "buckyball.bank_maxpool") {
        value = op->getOperand(1);
        continue;
      }
      if (name == "buckyball.bank_int8add") {
        value = op->getOperand(2);
        continue;
      }
      if (name == "buckyball.bank_int8mul") {
        value = op->getOperand(2);
        continue;
      }
      if (name == "buckyball.bank_smatmul" ||
          name == "buckyball.bank_vecmat16") {
        value = op->getOperand(2);
        continue;
      }
    }
    if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
      unsigned resultNumber = cast<OpResult>(value).getResultNumber();
      value = cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                  .getResults()[resultNumber];
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto forOp = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (!forOp || argument.getArgNumber() == 0)
        return std::nullopt;
      value = forOp.getInitArgs()[argument.getArgNumber() - 1];
      continue;
    }
    return std::nullopt;
  }
}

std::optional<BankSlot> PhysicalBankState::getSlot(Value value) const {
  auto bank = getConstI64(value);
  if (!bank)
    return std::nullopt;
  auto slot = vm.find(*bank);
  if (slot == vm.end())
    return std::nullopt;
  return slot->second;
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
    InFlightDiagnostic diagnostic =
        op->emitError("release of unknown virtual bank handle");
    diagnostic << " (bank=" << bank << ", live=" << vm.size()
               << ", used=" << getUsedCount() << "/" << bankNum << ")";
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
