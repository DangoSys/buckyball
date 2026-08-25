#ifndef BUCKYBALL_TARGET_REGISTRY_H
#define BUCKYBALL_TARGET_REGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace buckyball_target {

struct BuckyballIsaEntry {
  llvm::StringRef mnemonic;
  int32_t funct7;
};

struct BuckyballTargetConfig {
  llvm::StringRef name;
  llvm::StringRef core;
  int64_t bankNum;
  int64_t bankWidthBits;
  int64_t bankDepth;
  llvm::ArrayRef<llvm::StringRef> balls;
  llvm::ArrayRef<BuckyballIsaEntry> isa;
};

const BuckyballTargetConfig &getBuckyballTarget();
int32_t getBuckyballFunct7(llvm::StringRef mnemonic);
void requireBuckyballBall(llvm::StringRef ballName);

} // namespace buckyball_target

#endif // BUCKYBALL_TARGET_REGISTRY_H
