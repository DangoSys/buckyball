//===- LowerBuckyball.h - Buckyball lowering hooks -------------*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWER_BUCKYBALL_LOWER_BUCKYBALL_H
#define BUDDY_CONVERSION_LOWER_BUCKYBALL_LOWER_BUCKYBALL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;

namespace buddy {

struct BankSlot {
  int64_t base = -1;
  int64_t row = 1;
  int64_t col = 1;
  llvm::SmallVector<int64_t, 4> physicalGroups;
};

class PhysicalBankState {
public:
  PhysicalBankState(int64_t bankNum, int64_t privateBankMax);

  int64_t getBankNum() const { return bankNum; }
  int64_t getPrivateBankMax() const { return privateBankMax; }
  int64_t getUsedCount() const;
  bool empty() const { return vm.empty(); }

  std::optional<int64_t> getConstI64(Value value) const;
  std::optional<BankSlot> getSlot(Value value) const;
  std::optional<int64_t> tryAlloc(int64_t row, int64_t col);
  LogicalResult verifyKernelHandles(Operation *op, Value read,
                                    Value write) const;
  LogicalResult transfer(Operation *op, int64_t source, int64_t target);
  LogicalResult release(Operation *op, int64_t bank);

  Value cstI64(OpBuilder &builder, Location loc, uint64_t value) const;
  void createMset(OpBuilder &builder, Location loc, uint64_t bankId, bool alloc,
                  uint64_t row, uint64_t col) const;

private:
  void freeAlloc(const BankSlot &slot);

  int64_t bankNum = 0;
  int64_t privateBankMax = 0;
  llvm::DenseMap<int64_t, BankSlot> vm;
  llvm::SmallVector<int8_t, 32> used;
};

LogicalResult verifyNoBankSSAOps(Operation *root);
void addBaseAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                       PhysicalBankState &state);
// Lowers the operations implementing BankSSALowering; for design passes that
// lower some operations themselves before the rest.
void populateBankSSALoweringPatterns(RewritePatternSet &patterns);
void registerAssignPhysicalBanksPass();
// The single entry the compiler driver calls to register every pass.
void registerBuckyballPasses();
void registerLowerBuckyballToBankSSAPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWER_BUCKYBALL_LOWER_BUCKYBALL_H
