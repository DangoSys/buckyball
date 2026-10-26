//===- PhysicalBankState.cpp - Physical bank allocation state -------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::buddy;
using namespace ::buddy::buckyball;

PhysicalBankState::PhysicalBankState(int64_t bankNum, int64_t privateBankMax)
    : bankNum(bankNum), privateBankMax(privateBankMax), used(bankNum, 0) {}

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
    if (auto op = value.getDefiningOp<BankMvin2dOp>()) {
      value = op.getBank();
      continue;
    }
    if (auto op = value.getDefiningOp<BankMvoutOp>()) {
      value = op.getBank();
      continue;
    }
    if (auto op = value.getDefiningOp<BankTransferOp>()) {
      value = op.getTarget();
      continue;
    }
    if (auto op = value.getDefiningOp<BankKernelOp>()) {
      value = cast<OpResult>(value).getResultNumber() == 0 ? op.getReadBank()
                                                           : op.getWriteBank();
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
      if (name == "buckyball.bank_smatmul" || name == "buckyball.bank_mxfp8" ||
          name == "buckyball.bank_vecmat16") {
        value = op->getOperand(2);
        continue;
      }
      if (name == "buckyball.bank_gemmini_preload") {
        value = op->getOperand(1);
        continue;
      }
      if (name == "buckyball.bank_gemmini_compute_preloaded" ||
          name == "buckyball.bank_gemmini_compute_accumulated") {
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
    if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
      unsigned resultNumber = cast<OpResult>(value).getResultNumber();
      auto thenYield = cast<scf::YieldOp>(ifOp.getThenRegion().front().back());
      auto elseYield = cast<scf::YieldOp>(ifOp.getElseRegion().front().back());
      auto thenBank = getConstI64(thenYield.getResults()[resultNumber]);
      auto elseBank = getConstI64(elseYield.getResults()[resultNumber]);
      if (!thenBank || !elseBank || *thenBank != *elseBank)
        return std::nullopt;
      return thenBank;
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
  for (auto [index, group] : llvm::enumerate(slot->second.physicalGroups))
    if (group != slot->second.base + static_cast<int64_t>(index)) {
      emitError(value.getLoc(),
                "bank operation requires contiguous physical groups");
      return std::nullopt;
    }
  return slot->second;
}

std::optional<int64_t> PhysicalBankState::tryAlloc(int64_t row, int64_t col) {
  if (row > bankNum / col || row * col > bankNum - getUsedCount())
    return std::nullopt;
  int64_t bank = 0;
  while (bank <= privateBankMax && vm.contains(bank))
    ++bank;
  if (bank > privateBankMax)
    return std::nullopt;
  BankSlot slot;
  slot.row = row;
  slot.col = col;
  for (int64_t group = 0;
       group < bankNum &&
       static_cast<int64_t>(slot.physicalGroups.size()) < row * col;
       ++group)
    if (!used[group]) {
      slot.physicalGroups.push_back(group);
      used[group] = 1;
    }
  slot.base = slot.physicalGroups.front();
  vm[bank] = std::move(slot);
  return bank;
}

LogicalResult PhysicalBankState::verifyKernelHandles(Operation *op, Value read,
                                                     Value write) const {
  auto r = getConstI64(read), w = getConstI64(write);
  if (!r || !w || *r == *w || !vm.contains(*r) || !vm.contains(*w))
    return op->emitError(
        "bank kernel requires distinct live read/write handles");
  return success();
}

LogicalResult PhysicalBankState::transfer(Operation *op, int64_t source,
                                          int64_t target) {
  if (source == target)
    return op->emitError("bank transfer source and target must differ");
  if (target < 0 || target > privateBankMax)
    return op->emitError(
        "bank transfer target exceeds the private virtual ID range");
  auto sourceSlot = vm.find(source);
  if (sourceSlot == vm.end())
    return op->emitError("transfer consumes an unknown virtual bank handle");
  BankSlot moved = sourceSlot->second;
  vm.erase(sourceSlot);
  auto targetSlot = vm.find(target);
  if (targetSlot == vm.end()) {
    vm[target] = std::move(moved);
  } else {
    auto &slot = targetSlot->second;
    slot.physicalGroups.append(moved.physicalGroups);
    slot.row = 1;
    slot.col = slot.physicalGroups.size();
  }
  return success();
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
  for (int64_t group : slot.physicalGroups)
    used[group] = 0;
}
