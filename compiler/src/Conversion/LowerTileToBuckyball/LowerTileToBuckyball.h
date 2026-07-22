//===- LowerTileToBuckyball.h - Tile to Buckyball conversion hooks-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWER_TILE_TO_BUCKYBALL_H
#define BUDDY_CONVERSION_LOWER_TILE_TO_BUCKYBALL_H

#include "mlir/IR/PatternMatch.h"

#include <cstdint>

namespace mlir {
namespace buddy {

inline constexpr int64_t kDefaultBankWidthBytes = 16;
inline constexpr size_t kMatmulTile = 16;
inline constexpr size_t kMaxAccMvoutDepthLines = 256;
inline constexpr size_t kMaxI8MvinDepthLines = 1024;

inline size_t ceilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

inline size_t aMvinDepthLines(size_t mEl, size_t kEl) {
  return mEl * (kEl / 16);
}

inline size_t bMvinDepthLines(size_t kEl, size_t nEl) {
  return kEl * (nEl / 16);
}

inline size_t cMvoutDepthLines(size_t mEl, size_t nEl) {
  return mEl * (nEl / 16);
}

void populateMatrixTileMatMulPatterns(RewritePatternSet &patterns,
                                      int64_t bankWidthBytes, int64_t bankDepth,
                                      int64_t bankNum);

void registerLowerTileToBuckyballPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWER_TILE_TO_BUCKYBALL_H
