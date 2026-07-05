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
};

class PhysicalBankState {
public:
  explicit PhysicalBankState(int64_t bankNum);

  int64_t getBankNum() const { return bankNum; }
  bool empty() const { return vm.empty(); }

  std::optional<int64_t> getConstI64(Value value) const;
  std::optional<int64_t> tryAlloc(int64_t row, int64_t col);
  LogicalResult release(Operation *op, int64_t bank);

  void remember(int64_t bank, int64_t row, int64_t col);
  Value cstI64(OpBuilder &builder, Location loc, uint64_t value) const;
  void createMset(OpBuilder &builder, Location loc, uint64_t bankId, bool alloc,
                  uint64_t row, uint64_t col) const;

private:
  void freeAlloc(const BankSlot &slot);

  int64_t bankNum = 0;
  llvm::DenseMap<int64_t, BankSlot> vm;
  llvm::SmallVector<int8_t, 32> used;
};

LogicalResult verifyNoBankSSAOps(Operation *root);
void addBaseAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                       PhysicalBankState &state);
void registerAssignPhysicalBanksPass();
void registerLowerBankSSAToIntrinsicsPass();
void registerLowerBuckyballPass();
void registerLowerBuckyballToBankSSAPass();
void registerReportBankUsagePass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWER_BUCKYBALL_LOWER_BUCKYBALL_H
