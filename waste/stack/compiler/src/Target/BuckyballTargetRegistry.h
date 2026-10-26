#ifndef BUCKYBALL_TARGET_REGISTRY_H
#define BUCKYBALL_TARGET_REGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace buckyball_target {

struct BuckyballIsaEntry {
  llvm::StringRef mnemonic;
  int32_t funct7;
  unsigned ballIndex;
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
  llvm::StringRef core;
  int64_t bankNum;
  int64_t bankWidthBits;
  int64_t bankDepth;
  int64_t privateBankMax;
  bool rvvEnabled;
  llvm::ArrayRef<BuckyballBallMapping> ballMappings;
  llvm::ArrayRef<BuckyballIsaEntry> isa;
};

const BuckyballTargetConfig &getBuckyballTarget();
int32_t getBuckyballFunct7(llvm::StringRef mnemonic);
const BuckyballBallMapping &getBuckyballBallMapping(llvm::StringRef mnemonic);
int64_t getBuckyballBallParam(llvm::StringRef mnemonic, llvm::StringRef param);

} // namespace buckyball_target

#endif // BUCKYBALL_TARGET_REGISTRY_H
