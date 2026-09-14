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

struct BuckyballBallParam {
  llvm::StringRef name;
  int64_t value;
};

struct BuckyballBallMapping {
  llvm::StringRef name;
  int64_t inBW;
  int64_t outBW;
  llvm::ArrayRef<BuckyballBallParam> params;
};

struct BuckyballTargetConfig {
  llvm::StringRef name;
  llvm::StringRef core;
  int64_t bankNum;
  int64_t bankWidthBits;
  int64_t bankDepth;
  llvm::ArrayRef<llvm::StringRef> balls;
  llvm::ArrayRef<BuckyballBallMapping> ballMappings;
  llvm::ArrayRef<BuckyballIsaEntry> isa;
};

llvm::StringRef getRequestedBuckyballTarget();
const BuckyballTargetConfig &getBuckyballTarget();
int32_t getBuckyballFunct7(llvm::StringRef mnemonic);
void requireBuckyballBall(llvm::StringRef ballName);
const BuckyballBallMapping &getBuckyballBallMapping(llvm::StringRef ballName);
int64_t getBuckyballBallParam(llvm::StringRef ballName, llvm::StringRef param);

} // namespace buckyball_target

#endif // BUCKYBALL_TARGET_REGISTRY_H
